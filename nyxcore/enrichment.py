"""Explicit, read-only requests to MusicBrainz and Cover Art Archive."""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import re
import socket
import stat
import tempfile
import time
from collections.abc import Callable
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit

import aiohttp
from aiohttp.abc import AbstractResolver

from nyxcore.core.text import normalize_match_text

MB = "musicbrainz.org"
CAA = "coverartarchive.org"
ARCHIVE_MIRROR = re.compile(r"(?:(?:ia|dn)[0-9]{5,9}(?:\.(?:us|ca))?|s3\.us)\.archive\.org\Z")
MBID = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
IMAGE_ID = r"[0-9]{1,20}"
MAX_ART_BYTES = 6 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
CACHE_BYTES = 64 * 1024 * 1024
CACHE_FILES = 160
RECORDING_LIMIT = 25
RELEASE_PAGE_SIZE = 25
RELEASE_PAGES = 4
USER_AGENT = "NyxCore/0.3.0 (https://github.com/DRGYZ/nyxcore-ai-audio/issues)"
CACHE_LOCK = Lock()
VERSION_PATTERN = re.compile(r"\b(live|demo|instrumental|remix|edit|acoustic|karaoke|studio|video|mix|orchestral|single version)\b", re.I)
TITLE_QUALIFIER = re.compile(r"(?:\s+[-–—]\s+([^()\[\]]+)|\s*[\[(]([^()\[\]]+)[)\]])\s*$")
BILINGUAL_TITLE = re.compile(r"\s+/([^/\\]+?)/\s*$")


class ProviderError(ValueError):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def approved_url(url: str, *, artwork: bool) -> str:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if parsed.username is not None or parsed.password is not None or parsed.port not in {None, 443}:
            raise ValueError
        if artwork:
            approved = host in {CAA, "archive.org", "www.archive.org"} or bool(ARCHIVE_MIRROR.fullmatch(host or ""))
        else:
            approved = host == MB
        if not approved or parsed.scheme not in ({"https", "http"} if artwork else {"https"}):
            raise ValueError
        if parsed.netloc.lower() not in {host, f"{host}:443"}:
            raise ValueError
        # The archive API still advertises some HTTP URLs. Always use TLS before connecting.
        return urlunsplit(("https", host, parsed.path, parsed.query, ""))
    except (TypeError, ValueError) as exc:
        raise ProviderError("The provider returned an unapproved address.") from exc


def is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if ip.version == 6 and (ip.is_site_local or ip.sixtofour or ip.teredo or ip in ipaddress.ip_network("64:ff9b::/96")):
        return False
    return ip.is_global and not (ip.is_multicast or ip.is_reserved)


class PublicResolver(AbstractResolver):
    """Validate the addresses aiohttp actually uses for the TLS connection."""

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_UNSPEC):
        addresses = await asyncio.get_running_loop().getaddrinfo(
            host, port, family=family, type=socket.SOCK_STREAM,
        )
        if not addresses or any(not is_public_address(item[4][0]) for item in addresses):
            raise OSError("Provider did not resolve exclusively to public unicast addresses.")
        return [
            {"hostname": host, "host": address[0], "port": address[1], "family": af,
             "proto": proto, "flags": socket.AI_NUMERICHOST}
            for af, _, proto, _, address in addresses
        ]

    async def close(self):
        pass


class Transport:
    """Limit requests globally within the single supported server process."""

    def __init__(self):
        self._mb = asyncio.Lock()
        self._images = asyncio.Lock()
        self._next_request = 0.0

    async def get(self, url: str, *, artwork: bool, maximum: int) -> bytes | None:
        gate = self._images if artwork else self._mb
        if gate.locked():
            raise ProviderError("An online lookup is already running. Try again shortly.", 429)
        async with gate:
            try:
                async with asyncio.timeout(16):
                    connector = aiohttp.TCPConnector(resolver=PublicResolver(), use_dns_cache=False, limit=1)
                    async with aiohttp.ClientSession(
                        connector=connector, trust_env=False, auto_decompress=False,
                        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
                        timeout=aiohttp.ClientTimeout(total=6, connect=4, sock_read=4),
                    ) as session:
                        for attempt in range(2):
                            if not artwork:
                                wait = self._next_request - time.monotonic()
                                if wait > 3:
                                    raise ProviderError("MusicBrainz asked clients to back off. Try again later.", 503)
                                await asyncio.sleep(max(0, wait))
                                self._next_request = time.monotonic() + 1.05
                            result = await self._request(session, url, artwork, maximum)
                            if not isinstance(result, int):
                                return result
                            if not artwork:
                                self._next_request = max(self._next_request, time.monotonic() + result)
                            if result > 3 or attempt:
                                break
                            await asyncio.sleep(result)
            except (OSError, aiohttp.ClientError, TimeoutError) as exc:
                raise ProviderError("The online provider could not be reached. Manual editing is still available.", 503) from exc
        raise ProviderError("The online provider is busy. Wait a moment before trying again.", 503)

    async def _request(self, session, url: str, artwork: bool, maximum: int) -> bytes | int | None:
        for _ in range(5):
            url = approved_url(url, artwork=artwork)
            async with session.get(url, allow_redirects=False) as response:
                if response.status in {301, 302, 303, 307, 308} and artwork:
                    if not (location := response.headers.get("Location")):
                        break
                    url = urljoin(url, location)
                    continue
                if response.status == 404:
                    return None
                if response.status in {429, 503}:
                    value = response.headers.get("Retry-After", "").strip()
                    if value.isascii() and value.isdecimal() and len(value) < 6:
                        return max(1, min(int(value), 600))
                    try:
                        after = parsedate_to_datetime(value).timestamp() - time.time()
                        return max(1, min(int(after) + 1, 600))
                    except (TypeError, ValueError, OverflowError, OSError):
                        return 2
                if response.status != 200:
                    raise ProviderError(f"The online provider returned HTTP {response.status}.")
                encoding = response.headers.get("Content-Encoding", "identity")
                if encoding.lower() != "identity":
                    raise ProviderError("The provider returned an unsupported transfer encoding.")
                if response.content_length is not None and response.content_length > maximum:
                    raise ProviderError("The provider response exceeds the download limit.")
                body = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    body.extend(chunk)
                    if len(body) > maximum:
                        raise ProviderError("The provider response exceeds the download limit.")
                return bytes(body)
        raise ProviderError("The artwork link has an invalid redirect chain.")


def _linked(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


class WorkspaceCache:
    def __init__(self, workspace: Callable[[], Path]):
        self.workspace = workspace

    def _directory(self) -> Path:
        parent = self.workspace().resolve()
        cache = parent / "music-info-cache"
        cache.mkdir(parents=True, exist_ok=True)
        if _linked(cache) or not cache.is_dir():
            raise ProviderError("Music Info needs a regular cache directory inside the workspace.", 409)
        return cache

    def _path(self, purpose: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._directory() / f"{purpose}-{digest}.cache"

    def read(self, purpose: str, key: str, *, age: int, maximum: int) -> bytes | None:
        with CACHE_LOCK:
            try:
                path = self._path(purpose, key)
                if _linked(path) or not path.is_file() or time.time() - path.stat().st_mtime > age:
                    return None
                with path.open("rb") as stream:
                    value = stream.read(maximum + 1)
                return value if len(value) <= maximum else None
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise ProviderError("Music Info could not read its workspace cache.", 409) from exc

    def write(self, purpose: str, key: str, data: bytes) -> None:
        with CACHE_LOCK:
            try:
                path = self._path(purpose, key)
                if len(data) > CACHE_BYTES:
                    raise ProviderError("This item is too large for the workspace cache.")
                entries = []
                for cached in path.parent.glob("*.cache"):
                    generated = re.fullmatch(r"(response|thumbnail|cover-index|approved)-[0-9a-f]{64}\.cache", cached.name)
                    if generated and cached != path and not _linked(cached) and cached.is_file():
                        info = cached.stat()
                        entries.append((info.st_mtime, info.st_size, cached))
                size = sum(item[1] for item in entries)
                entries.sort()
                for index, (_, length, cached) in enumerate(entries):
                    if size + len(data) <= CACHE_BYTES and len(entries) - index < CACHE_FILES:
                        break
                    cached.unlink()
                    size -= length
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    temporary.write_bytes(data)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            except OSError as exc:
                raise ProviderError("Music Info could not save to the workspace cache.", 409) from exc


def mbid(value: object) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(MBID, value) else None


def text(value: object, maximum: int = 512) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(character for character in value[:maximum] if character.isprintable()).strip()


def objects(value: object, maximum: int = 50) -> list[dict]:
    return [item for item in value[:maximum] if isinstance(item, dict)] if isinstance(value, list) else []


def release_choices(items: list[dict]) -> list[dict]:
    result: dict[tuple[str, str], dict] = {}
    for release in items:
        identifier, title = mbid(release.get("id")), text(release.get("title"))
        group = release.get("release-group")
        group = group if isinstance(group, dict) else {}
        group_id, group_title = mbid(group.get("id")), text(group.get("title"))
        if identifier and title:
            result[("release", identifier)] = {
                "id": identifier, "kind": "release", "title": title, "date": text(release.get("date"), 10),
                "country": text(release.get("country"), 12), "group_id": group_id,
            }
        if identifier and title and group_id and group_title:
            result[("release-group", group_id)] = {
                "id": group_id, "kind": "release-group", "title": group_title,
                "date": text(group.get("first-release-date"), 10), "country": "", "group_id": group_id,
            }
    return list(result.values())


def title_names(title: str) -> list[str]:
    """Keep alternative scripts and a plain title for local textual comparison."""
    bilingual = BILINGUAL_TITLE.search(title)
    if bilingual:
        names = [title[:bilingual.start()], bilingual[1]]
    else:
        names = re.split(r"\s+[=/]\s+", title)
    return list(dict.fromkeys(name.strip() for name in names if name.strip()))


def similarity(left: str, right: str) -> float:
    a, b = normalize_match_text(left), normalize_match_text(right)
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0


def release_similarity(album: str, name: str) -> float:
    # A singles collection can be tagged with its subtitle while the catalog uses its short title.
    strip_subtitle = r"\s*[(\[]\s*(?:the\s+)?singles collection\b.*$"
    return similarity(re.sub(strip_subtitle, "", album, flags=re.I), re.sub(strip_subtitle, "", name, flags=re.I))


def version_in(label: str) -> str | None:
    match = VERSION_PATTERN.search(label)
    if not match:
        return None
    return {"mix": "remix"}.get(match[1].lower(), match[1].lower())


def title_version(title: str) -> str | None:
    qualifier = TITLE_QUALIFIER.search(title)
    return version_in(qualifier[1] or qualifier[2]) if qualifier else None


def standard_album(release: dict) -> bool:
    group = release.get("release-group")
    return (isinstance(group, dict) and group.get("primary-type") == "Album"
            and not group.get("secondary-types") and release.get("status") == "Official")


def recording_version(row: dict, name: str, detail: str, items: list[dict]) -> str | None:
    media = "video" if row.get("video") is True else None
    if version := version_in(detail) or title_version(name) or media:
        return version
    # Only infer from a release if every supplied release points to the same alternate version.
    release_versions = set()
    for item in items:
        group = item.get("release-group")
        secondary = group.get("secondary-types", []) if isinstance(group, dict) else []
        album = text(item.get("title"))
        release_versions.add(
            "live" if isinstance(secondary, list) and "Live" in secondary
            else title_version(album) or (
                version_in(album) if re.match(r"(?i)^(?:live\b|.*\b(?:instrumentals|demos)\b)", album) else None
            )
        )
    return next(iter(release_versions)) if len(release_versions) == 1 else None


def title_similarity(query: str, candidate: str) -> float:
    best = 0.0
    for left in title_names(query):
        for right in title_names(candidate):
            best = max(best, similarity(left, right))
            # A subtitle can describe a version or a movement. Show the difference in identity evidence.
            plain_left = TITLE_QUALIFIER.sub("", left)
            plain_right = TITLE_QUALIFIER.sub("", right)
            if plain_left != left or plain_right != right:
                best = max(best, 0.97 * similarity(plain_left, plain_right))
    return best


def recording_candidates(
    payload: dict, title: str, artist: str, album: str, duration_ms: int | None,
    filename_title: str = "", filename_artist: str = "", version_hint: str = "", version_detail: str = "",
) -> list[dict]:
    result = []
    expected_version = version_hint or title_version(title)
    qualifier_words = set(normalize_match_text(version_detail).split())
    qualifier_words -= {"live", "in", "at", "from", "the", "version", "remix", "mix", "edit", "demo",
                        "instrumental", "acoustic", "karaoke", "studio", "video", "orchestral", "single"}
    for recording in objects(payload.get("recordings"), RECORDING_LIMIT):
        identifier, name = mbid(recording.get("id")), text(recording.get("title"))
        if not identifier or not name:
            continue
        credits = objects(recording.get("artist-credit"), 12)
        names, artists = [], []
        for credit in credits:
            raw = credit.get("artist")
            raw = raw if isinstance(raw, dict) else {}
            credit_name = text(credit.get("name") or raw.get("name"))
            if not credit_name:
                continue
            names.append(credit_name + (credit.get("joinphrase", "")[:30] if isinstance(credit.get("joinphrase"), str) else ""))
            if artist_id := mbid(raw.get("id")):
                artists.append({"id": artist_id, "name": credit_name})
        credited = text("".join(names))
        items = objects(recording.get("releases"), 100)
        choices = release_choices(items)
        if album:
            choices.sort(key=lambda choice: -release_similarity(album, choice["title"]))
        detail = text(recording.get("disambiguation"), 180)
        version = recording_version(recording, name, detail, items)
        context_agrees = bool(qualifier_words and version == expected_version and any(
            qualifier_words <= set(normalize_match_text(context).split())
            for context in (name, detail, *(text(item.get("title")) for item in items))
        ))
        length = recording.get("length")
        length = length if type(length) is int and length > 0 else None
        evidence, weighted = {}, []
        for label, query, ratio, weight, available in (
            ("title", title, title_similarity(title, name), 5, True),
            ("artist", artist, similarity(artist, credited), 4, bool(credited)),
        ):
            if query and available:
                weighted.append((ratio, weight))
                evidence[label] = "agrees" if ratio >= 0.92 else "similar" if ratio >= 0.68 else "differs"
            else:
                evidence[label] = "unavailable"
        text_score = round(100 * sum(value * weight for value, weight in weighted)
                           / max(1, sum(weight for _, weight in weighted)))
        album_score = release_similarity(album, choices[0]["title"]) if album and choices else 0
        evidence["album"] = ("agrees" if album_score >= 0.92 else "similar" if album_score >= 0.68 else
                             "differs" if album and choices else "unavailable")
        delta = abs(duration_ms - length) if duration_ms and length else None
        if delta is not None:
            evidence["duration"] = "agrees" if delta <= 2500 else "similar" if delta <= 8000 else "differs"
        else:
            evidence["duration"] = "unavailable"
        filename_scores = [
            similarity(hint, candidate) for hint, candidate in
            ((filename_title, name), (filename_artist, credited)) if hint and candidate
        ]
        if filename_scores:
            ratio = sum(filename_scores) / len(filename_scores)
            evidence["filename"] = "agrees" if ratio >= 0.92 else "similar" if ratio >= 0.68 else "differs"
        else:
            ratio = 0
            evidence["filename"] = "unavailable"
        duration_points = {"agrees": 0, "similar": -8, "unavailable": -13, "differs": -25}[evidence["duration"]]
        if expected_version:
            version_points = 8 if version == expected_version else -9 if version is None else -20
        else:
            version_points = -11 if version and version != "studio" else 0
        identity = ("conflicting" if evidence["duration"] == "differs" or
                    expected_version and version and version != expected_version
                    else "supported" if evidence["duration"] == "agrees" and
                    (version == expected_version or version == "studio" and not expected_version) and
                    (not qualifier_words or context_agrees)
                    else "uncertain")
        album_points = 5 if not version and any(standard_album(item) for item in items) else 0
        context_points = 12 if context_agrees else 0
        score = min(100, max(0, round(0.75 * text_score + 9 * album_score + 3 * ratio)
                             + duration_points + version_points + album_points + context_points))
        result.append({
            "id": identifier, "title": name, "artist": credited, "artists": artists, "duration_ms": length,
            "releases": choices[:16], "evidence": evidence, "score": score, "text_score": text_score,
            "duration_delta_ms": delta, "version": version, "disambiguation": detail, "identity": identity,
            "version_context_agrees": context_agrees,
        })
    return sorted(result, key=lambda item: -item["score"])


TRANSPORT = Transport()


class Providers:
    def __init__(self, cache: WorkspaceCache, transport: Transport | None = None):
        self.cache = cache
        self.transport = transport or TRANSPORT

    async def _json(self, url: str, *, artwork: bool) -> dict:
        payload = self.cache.read("response", url, age=7 * 86400, maximum=MAX_JSON_BYTES)
        if payload is None:
            payload = await self.transport.get(url, artwork=artwork, maximum=MAX_JSON_BYTES)
            if payload is None:
                self.cache.write("response", url, b"{}")
                return {}
        try:
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise ValueError
        except (UnicodeError, ValueError) as exc:
            raise ProviderError("The online provider returned invalid data.") from exc
        # Cache successful responses and no-matches. Failed requests never overwrite drafts.
        if self.cache.read("response", url, age=7 * 86400, maximum=MAX_JSON_BYTES) is None:
            self.cache.write("response", url, payload)
        return parsed

    async def search(
        self, title: str, artist: str, album: str, duration_ms: int | None,
        filename_title: str = "", filename_artist: str = "", version_hint: str = "", version_detail: str = "",
    ) -> list[dict]:
        def quoted(value: str) -> str:
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

        async def recordings(parts: list[str]) -> dict:
            params = urlencode({"query": " AND ".join(parts), "fmt": "json", "limit": RECORDING_LIMIT})
            return await self._json(f"https://{MB}/ws/2/recording?{params}", artwork=False)

        def rank(payload: dict) -> list[dict]:
            return recording_candidates(payload, title, artist, album, duration_ms, filename_title, filename_artist,
                                        version_hint, version_detail)

        names = title_names(title)[:3]
        titles = [f"recording:{quoted(name)}" for name in names]
        credit = [f"artist:{quoted(artist)}"] if artist else []
        phrase = ["(" + " OR ".join(titles) + ")"] if len(titles) > 1 else titles
        primary = await recordings(phrase + credit + (["status:official"] if title and artist else []))
        rows = rank(primary)
        if not (title and artist) or (rows and rows[0]["text_score"] >= 88 and rows[0]["identity"] == "supported"):
            return rows[:12]

        # Drop the official-release filter once. Relax title punctuation only when the text itself needs help.
        fallback = phrase + credit
        if not rows or rows[0]["text_score"] < 88 or any(TITLE_QUALIFIER.search(name) for name in names):
            for name in names:
                base = TITLE_QUALIFIER.sub("", name).strip() or name
                terms = re.findall(r"[^\W_]+", base, re.UNICODE)
                if 2 <= len(terms) <= 12:
                    titles.append("recording:(" + " AND ".join(quoted(term) for term in terms) + ")")
                elif base != name:
                    titles.append(f"recording:{quoted(base)}")
            fallback = ["(" + " OR ".join(dict.fromkeys(titles)) + ")"] + credit
        try:
            extra = await recordings(fallback)
        except ProviderError as exc:
            if rows and exc.status in {429, 503}:
                return rows[:12]
            raise
        combined = {row["id"]: row for row in rows}
        for row in rank(extra):
            if row["id"] not in combined or row["score"] > combined[row["id"]]["score"]:
                combined[row["id"]] = row
        return sorted(combined.values(), key=lambda item: -item["score"])[:12]

    async def linked_releases(self, recording_id: str, page: int) -> tuple[list[dict], bool]:
        offset = (page - 1) * RELEASE_PAGE_SIZE
        params = urlencode({"recording": recording_id, "inc": "release-groups", "fmt": "json",
                            "limit": RELEASE_PAGE_SIZE, "offset": offset})
        payload = await self._json(f"https://{MB}/ws/2/release?{params}", artwork=False)
        items = objects(payload.get("releases"), RELEASE_PAGE_SIZE)
        total = payload.get("release-count")
        more = type(total) is int and offset + RELEASE_PAGE_SIZE < total and page < RELEASE_PAGES
        return release_choices(items), more

    async def cover_list(self, kind: str, identifier: str) -> list[dict]:
        payload = await self._json(f"https://{CAA}/{kind}/{identifier}/", artwork=True)
        source = re.fullmatch(r"https?://musicbrainz\.org/release/(" + MBID + ")/?", text(payload.get("release")))
        if not source or (kind == "release" and source[1] != identifier):
            return []
        choices = []
        for row in objects(payload.get("images"), 24):
            image = str(row.get("id", ""))
            types = row.get("types")
            if (row.get("approved") is True and re.fullmatch(IMAGE_ID, image)
                    and (row.get("front") is True or isinstance(types, list) and "Front" in types)):
                # Derive only a known suffix. Never fetch a URL supplied in JSON.
                path = text(row.get("image"))
                suffix = re.fullmatch(
                    rf"https?://coverartarchive\.org/release/{source[1]}/{image}(\.jpg|\.jpeg|\.png)",
                    path, re.IGNORECASE,
                )
                choices.append({"id": image, "release_id": source[1], "label": text(row.get("comment"), 100),
                                "extension": suffix[1].lower() if suffix else ""})
        return list({item["id"]: item for item in choices}.values())[:4]

    async def image(self, release: str, image: str, *, thumbnail: bool, extension: str = "") -> bytes:
        suffix = "-250.jpg" if thumbnail else extension
        url = f"https://{CAA}/release/{release}/{image}{suffix}"
        data = await self.transport.get(url, artwork=True, maximum=512 * 1024 if thumbnail else MAX_ART_BYTES)
        if not data:
            raise ProviderError("This cover image is unavailable.", 404)
        return data
