"""Keep a reviewed move rooted in checked parent directories."""
from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from nyxcore.core.filesystem import (
    FileIdentity, check_move_path, descriptor_mount_id, is_linked_stat, path_mount_id, portable_name_key,
)


def _identity(info: os.stat_result) -> FileIdentity:
    return info.st_dev, info.st_ino


def _windows_handle(path: Path, access: int, flags: int, stack: ExitStack, *, share: int = 1):
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    # Open the entry itself, including reparse points.
    handle = create(str(path), access, share, None, 3, flags | 0x00200000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    stack.callback(close, handle)
    return kernel, handle


def _unlink_windows_source(path: Path, identity: FileIdentity, expected_hash: str) -> None:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    with ExitStack() as stack:
        kernel, handle = _windows_handle(path, 0x80010000, 0, stack)  # GENERIC_READ | DELETE
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        stack.pop_all()  # The descriptor now owns the handle.
        with os.fdopen(descriptor, "rb") as stream:
            node = os.fstat(stream.fileno())
            if (not stat.S_ISREG(node.st_mode) or is_linked_stat(node) or _identity(node) != identity
                    or node.st_nlink != 2 or hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash):
                raise RuntimeError("Source changed before guarded unlink.")
            disposition = ctypes.c_ubyte(1)
            set_info = kernel.SetFileInformationByHandle
            set_info.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD]
            set_info.restype = wintypes.BOOL
            if not set_info(handle, 4, ctypes.byref(disposition), ctypes.sizeof(disposition)):  # FileDispositionInfo
                raise ctypes.WinError(ctypes.get_last_error())


@dataclass(frozen=True)
class Directory:
    path: Path
    identity: FileIdentity
    fd: int | None
    mount: int | None

    def entry(self, name: str) -> os.stat_result | None:
        with os.scandir(self.path if self.fd is None else self.fd) as entries:
            matches = {entry.name for entry in entries if portable_name_key(entry.name) == portable_name_key(name)}
        if matches and matches != {name}:
            raise RuntimeError(f"Equivalent path spelling is occupied: {self.path / name}")
        try:
            if self.fd is None:
                info = (self.path / name).lstat()
            else:
                info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not matches or is_linked_stat(info) or info.st_dev != self.identity[0]:
            raise RuntimeError(f"Linked, aliased, or cross-filesystem path: {self.path / name}")
        if path_mount_id(name, dir_fd=self.fd) != self.mount:
            raise RuntimeError(f"Path crosses a mount boundary inside the library: {self.path / name}")
        return info

    def verify(self) -> None:
        info = self.path.lstat()
        if (is_linked_stat(info) or not stat.S_ISDIR(info.st_mode) or _identity(info) != self.identity):
            raise RuntimeError(f"Move directory changed: {self.path}")
        if path_mount_id(self.path) != self.mount:
            raise RuntimeError(f"Move directory crosses a mount boundary inside the library: {self.path}")


@dataclass(frozen=True)
class MovePaths:
    source: Path
    destination: Path
    source_parent: Directory
    destination_parent: Directory
    directories: tuple[Directory, ...]

    def _verify_directories(self) -> None:
        for directory in self.directories:
            directory.verify()

    def _verify_file(self, directory: Directory, name: str, identity: FileIdentity, links: int) -> None:
        entry = directory.entry(name)
        if entry is None or not stat.S_ISREG(entry.st_mode) or _identity(entry) != identity or entry.st_nlink != links:
            raise RuntimeError(f"File identity or link count changed: {directory.path / name}")

    def verify_before_link(self, identity: FileIdentity) -> None:
        self._verify_directories()
        if self.destination_parent.entry(self.destination.name) is not None:
            raise RuntimeError(f"Destination already exists: {self.destination}")
        self._verify_file(self.source_parent, self.source.name, identity, 1)

    def verify_linked(self, identity: FileIdentity) -> None:
        self._verify_directories()
        self._verify_file(self.destination_parent, self.destination.name, identity, 2)
        self._verify_file(self.source_parent, self.source.name, identity, 2)

    def verify_completed(self, identity: FileIdentity) -> None:
        self._verify_directories()
        if self.source_parent.entry(self.source.name) is not None:
            raise RuntimeError("The old path was occupied during move completion.")
        self._verify_file(self.destination_parent, self.destination.name, identity, 1)

    def link(self) -> None:
        self._verify_directories()
        if self.source_parent.fd is None:
            os.link(self.source, self.destination)
        else:
            if path_mount_id(self.source.name, dir_fd=self.source_parent.fd) != self.source_parent.mount:
                raise RuntimeError(f"Move source crosses a mount boundary inside the library: {self.source}")
            os.link(self.source.name, self.destination.name, src_dir_fd=self.source_parent.fd,
                    dst_dir_fd=self.destination_parent.fd, follow_symlinks=False)

    def unlink_source(self, identity: FileIdentity, expected_hash: str) -> None:
        self.verify_linked(identity)
        if self.source_parent.fd is None:
            _unlink_windows_source(self.source, identity, expected_hash)
        else:
            os.unlink(self.source.name, dir_fd=self.source_parent.fd)


@contextmanager
def move_scope(root: Path, source: Path, destination: Path, *, create_parents: bool = False) -> Iterator[MovePaths]:
    check_move_path(root, source)
    if check_move_path(root, destination, missing_ok=True) is not None and create_parents:
        raise RuntimeError(f"Destination already exists: {destination}")
    mount = path_mount_id(root)
    with ExitStack() as stack:
        directories: dict[Path, Directory] = {}

        def open_directory(path: Path, parent: Directory | None, info: os.stat_result) -> Directory:
            if not stat.S_ISDIR(info.st_mode) or is_linked_stat(info):
                raise RuntimeError(f"Invalid move directory: {path}")
            fd = None
            if os.name == "nt":
                # Listing access makes the share restrictions effective for directory renames.
                _windows_handle(path, 0x81, 0x02000000, stack, share=3)
            else:
                fd = os.open(path if parent is None else path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                             dir_fd=None if parent is None else parent.fd)
                stack.callback(os.close, fd)
                if _identity(os.fstat(fd)) != _identity(info):
                    raise RuntimeError(f"Move directory changed while opening: {path}")
                if descriptor_mount_id(fd) != mount:
                    raise RuntimeError(f"Move directory crosses a mount boundary inside the library: {path}")
            directory = Directory(path, _identity(info), fd, mount)
            directory.verify()
            directories[path] = directory
            return directory

        root_directory = open_directory(root, None, root.lstat())

        def parent_of(file: Path, create: bool) -> Directory:
            parent = root_directory
            for name in file.parent.relative_to(root).parts:
                path = parent.path / name
                if path in directories:
                    parent = directories[path]
                    continue
                parent.verify()
                info = parent.entry(name)
                if info is None and create:
                    if parent.fd is None:
                        path.mkdir()
                    else:
                        os.mkdir(name, dir_fd=parent.fd)
                    info = parent.entry(name)
                if info is None:
                    raise RuntimeError(f"Move folder is missing: {path}")
                parent = open_directory(path, parent, info)
            return parent

        source_parent = parent_of(source, False)
        destination_parent = parent_of(destination, create_parents)
        yield MovePaths(source, destination, source_parent, destination_parent, tuple(directories.values()))
