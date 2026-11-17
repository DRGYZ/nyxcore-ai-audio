from __future__ import annotations

import base64
import asyncio
import hashlib
import hmac
import io
import re
import unicodedata
from collections.abc import Callable
from typing import Literal

from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from nyxcore.enrichment import BILINGUAL_TITLE, IMAGE_ID, MAX_ART_BYTES, MBID, RELEASE_PAGES, ProviderError, Providers
from nyxcore.tagging.writer import ArtworkEdit

HEX = r"^[0-9a-f]{64}$"
MAX_PIXELS = 16_000_000
ART_AGE = 14 * 86400
RELEASE_TYPE = Literal["release", "release-group"]


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Lookup(RequestModel):
    title: str = Field("", max_length=160)
    artist: str = Field("", max_length=160)
    album: str = Field("", max_length=160)
    duration_ms: int | None = Field(None, gt=0, le=24 * 3600 * 1000)
    filename_title: str = Field("", max_length=160)
    filename_artist: str = Field("", max_length=160)
    version_hint: Literal["", "live", "demo", "instrumental", "remix", "edit", "acoustic", "karaoke",
                          "studio", "video", "orchestral", "single version"] = ""
    version_detail: str = Field("", max_length=100)

    @field_validator("title", "artist")
    @classmethod
    def validate_query(cls, value: str, info: ValidationInfo) -> str:
        path_check = value
        alternate = BILINGUAL_TITLE.search(value) if info.field_name == "title" else None
        if alternate:
            # Allow a trailing alternate-language title delimited by slashes.
            if any(char.isalpha() and not char.isascii() for char in alternate[1]) and "." not in alternate[1]:
                path_check = value[:alternate.start()]
        if (
            "\\" in value
            or re.search(r"(?:^|\s)(?:/(?!\s)|(?:~|\.{1,2})/|[A-Za-z]:/)|://", path_check)
            or re.search(r"/[^/\s]+\.(?:mp3|m4a|flac|wav|aac|ogg|opus)(?:\s|$)", value, re.I)
            or re.fullmatch(r"(?i)(?:[0-9a-f]{32}|[0-9a-f]{40}|[0-9a-f]{64})", value.strip())
            or any(unicodedata.category(char) in {"Cc", "Cs"} for char in value)
        ):
            raise ValueError("Enter music titles and artists without paths, URLs, or control characters.")
        return value.strip()


class ReleaseSource(RequestModel):
    kind: RELEASE_TYPE
    id: str = Field(pattern=f"^{MBID}$")
    title: str = Field(min_length=1, max_length=512)


class Source(RequestModel):
    recording_id: str = Field(pattern=f"^{MBID}$")
    title: str = Field(min_length=1, max_length=512)
    artist: str = Field(max_length=512)
    release: ReleaseSource | None = None
    token: str = Field(pattern=HEX)

    def tag_values(self) -> dict[str, str | None]:
        return {"title": self.title, "artist": self.artist, "album": self.release.title if self.release else None}


class ReleaseBrowse(RequestModel):
    source: Source
    page: int = Field(1, ge=1, le=RELEASE_PAGES)


class CoverChoice(RequestModel):
    id: str = Field(pattern=f"^{IMAGE_ID}$")
    release_id: str = Field(pattern=f"^{MBID}$")
    extension: Literal["", ".jpg", ".jpeg", ".png"] = ""
    token: str = Field(pattern=HEX)


class CoverSelection(RequestModel):
    source: Source
    choice: CoverChoice


class ArtworkReference(RequestModel):
    digest: str = Field(pattern=HEX)
    image_id: str = Field(pattern=f"^{IMAGE_ID}$")
    release_id: str = Field(pattern=f"^{MBID}$")
    width: int = Field(gt=0, le=6000)
    height: int = Field(gt=0, le=6000)
    token: str = Field(pattern=HEX)
    replace_existing: bool = False


def _image(data: bytes, *, thumb: bool) -> tuple[bytes, int, int]:
    """Decode a single bounded raster and emit a plain JPEG for browser/embed use."""
    if not data or len(data) > MAX_ART_BYTES:
        raise ProviderError("This cover exceeds the image size limit.")
    try:
        with Image.open(io.BytesIO(data), formats=("JPEG", "PNG")) as source:
            width, height = source.size
            if (
                width < 1 or height < 1 or width > 6000 or height > 6000
                or width * height > MAX_PIXELS or getattr(source, "n_frames", 1) != 1
            ):
                raise ProviderError("This cover exceeds the image dimensions limit or has multiple frames.")
            source.load()
            converted = ImageOps.exif_transpose(source)
            rgba = converted.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, "white")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
            width, height = rgb.size
            if thumb:
                rgb.thumbnail((224, 224))
            output = io.BytesIO()
            rgb.save(output, "JPEG", quality=78 if thumb else 90)
            cleaned = output.getvalue()
            if len(cleaned) > MAX_ART_BYTES:
                raise ProviderError("The normalized image exceeds the image size limit.")
            return cleaned, width, height
    except (Image.DecompressionBombError, OSError, ValueError) as exc:
        if isinstance(exc, ProviderError):
            raise
        raise ProviderError("The cover is not a readable single-frame JPEG or PNG.") from exc


def thumbnail(data: bytes | None) -> str | None:
    if data is None:
        return None
    try:
        validated, _, _ = _image(data, thumb=True)
    except ProviderError:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(validated).decode("ascii")


class Enrichment:
    def __init__(self, providers: Providers, proof: Callable[..., str]):
        self.providers = providers
        self.proof = proof

    def check_source(self, source: Source) -> None:
        expected = self.proof("source", source.model_dump(exclude={"token"}))
        if not hmac.compare_digest(source.token, expected):
            raise HTTPException(409, "The MusicBrainz selection is stale or changed. Select the recording or release again.")

    def _source(self, row: dict, release: dict | None = None) -> dict:
        value = {"recording_id": row["id"], "title": row["title"], "artist": row["artist"],
                 "release": {key: release[key] for key in ("id", "kind", "title")} if release else None}
        return {**value, "token": self.proof("source", value)}

    def _release_sources(self, row: dict, releases: list[dict]) -> list[dict]:
        return [{**release, "source": self._source(row, release)} for release in releases]

    def _artwork_proof(self, source: Source, fields: dict) -> str:
        return self.proof("artwork", source.token, fields)

    def cached_artwork(self, source: Source, selection: ArtworkReference) -> ArtworkEdit:
        fields = selection.model_dump(exclude={"token", "replace_existing"})
        if not source.release or not hmac.compare_digest(selection.token, self._artwork_proof(source, fields)):
            raise HTTPException(409, "This artwork is stale or belongs to a different release. Select it again.")
        try:
            data = self.providers.cache.read("approved", selection.digest, age=ART_AGE, maximum=MAX_ART_BYTES)
        except ProviderError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        if data is None or hashlib.sha256(data).hexdigest() != selection.digest:
            raise HTTPException(409, "The cached artwork is missing, expired, or changed. Select the cover again.")
        return ArtworkEdit(data, selection.digest, selection.width, selection.height, selection.replace_existing)

    def mount(self, app: FastAPI) -> None:
        @app.post("/api/metadata/lookup")
        async def lookup(query: Lookup) -> dict:
            if not query.title and not query.artist:
                raise HTTPException(422, "Enter a title or artist to search MusicBrainz.")
            try:
                rows = await self.providers.search(
                    query.title, query.artist, query.album, query.duration_ms, query.filename_title, query.filename_artist,
                    query.version_hint, query.version_detail,
                )
            except ProviderError as exc:
                raise HTTPException(exc.status, str(exc)) from exc
            for row in rows:
                row["source"] = self._source(row)
                row["releases"] = self._release_sources(row, row["releases"])
            uncertain = bool(rows and (
                rows[0]["text_score"] < 88 or rows[0]["identity"] != "supported" or not query.title or not query.artist
                or (len(rows) > 1 and rows[0]["score"] - rows[1]["score"] <= 8)
            ))
            return {"candidates": rows, "status": "ambiguous" if uncertain else "matches" if rows else "no_match"}

        @app.post("/api/metadata/releases")
        async def linked_releases(request: ReleaseBrowse) -> dict:
            source = request.source
            self.check_source(source)
            if source.release:
                raise HTTPException(422, "Select a recording before browsing its releases.")
            try:
                choices, more = await self.providers.linked_releases(source.recording_id, request.page)
            except ProviderError as exc:
                raise HTTPException(exc.status, str(exc)) from exc
            row = {"id": source.recording_id, "title": source.title, "artist": source.artist}
            return {"releases": self._release_sources(row, choices), "next_page": request.page + 1 if more else None}

        @app.post("/api/metadata/covers")
        async def covers(source: Source) -> dict:
            self.check_source(source)
            if not source.release:
                raise HTTPException(422, "Choose a release or release group first.")
            try:
                rows = await self.providers.cover_list(source.release.kind, source.release.id)
            except ProviderError as exc:
                raise HTTPException(exc.status, str(exc)) from exc
            for row in rows:
                row["token"] = self.proof("cover", source.token, row["release_id"], row["id"], row["extension"])
                row["thumbnail"] = None
            try:
                async with asyncio.timeout(12):
                    for row in rows:
                        cache_key = f"{row['release_id']}:{row['id']}:250"
                        try:
                            data = self.providers.cache.read("thumbnail", cache_key, age=ART_AGE, maximum=MAX_ART_BYTES)
                            if data is None:
                                original = await self.providers.image(row["release_id"], row["id"], thumbnail=True)
                                data, _, _ = await asyncio.to_thread(_image, original, thumb=True)
                                self.providers.cache.write("thumbnail", cache_key, data)
                            row["thumbnail"] = thumbnail(data)
                        except ProviderError:
                            continue
            except TimeoutError:
                pass  # A slow thumbnail does not hide an otherwise valid release choice.
            return {"covers": rows}

        @app.post("/api/metadata/cover-select")
        async def cover_select(request: CoverSelection) -> dict:
            source, choice = request.source, request.choice
            self.check_source(source)
            expected = self.proof("cover", source.token, choice.release_id, choice.id, choice.extension)
            if not source.release or not hmac.compare_digest(choice.token, expected):
                raise HTTPException(409, "This cover is not authorized for the selected release.")
            try:
                selection_key = f"{choice.release_id}:{choice.id}{choice.extension}"
                cached = self.providers.cache.read("cover-index", selection_key, age=ART_AGE, maximum=129)
                data = None
                if cached and re.fullmatch(rb"[0-9a-f]{64}:[0-9a-f]{64}", cached):
                    digest, receipt = cached.decode("ascii").split(":")
                    if hmac.compare_digest(receipt, self.proof("cover-cache", selection_key, digest)):
                        body = self.providers.cache.read("approved", digest, age=ART_AGE, maximum=MAX_ART_BYTES)
                        if body and hashlib.sha256(body).hexdigest() == digest:
                            with Image.open(io.BytesIO(body), formats=("JPEG",)) as image:
                                width, height = image.size
                            data = body
                if data is None:
                    original = await self.providers.image(choice.release_id, choice.id, thumbnail=False, extension=choice.extension)
                    data, width, height = await asyncio.to_thread(_image, original, thumb=False)
                    digest = hashlib.sha256(data).hexdigest()
                    self.providers.cache.write("approved", digest, data)
                    receipt = self.proof("cover-cache", selection_key, digest)
                    self.providers.cache.write("cover-index", selection_key, f"{digest}:{receipt}".encode("ascii"))
            except (OSError, ProviderError) as exc:
                raise HTTPException(getattr(exc, "status", 502), str(exc)) from exc
            fields = {"digest": digest, "image_id": choice.id, "release_id": choice.release_id,
                      "width": width, "height": height}
            return {"selection": {**fields, "token": self._artwork_proof(source, fields), "replace_existing": False},
                    "thumbnail": thumbnail(data)}
