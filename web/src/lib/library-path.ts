function absolutePathParts(path: string) {
  const drive = /^([a-z]):[\\/]/i.exec(path);
  if (drive) {
    return { volume: drive[1].toUpperCase(), parts: path.slice(3).split(/[\\/]/), separator: "\\" };
  }
  if (path.startsWith("/") && !path.startsWith("//")) {
    return { volume: "/", parts: path.slice(1).split("/"), separator: "/" };
  }
  return null;
}

// Display only. Decline to shorten paths with ambiguous components or a different root.
export function libraryRelativePath(path: string, root?: string) {
  if (!root) return null;
  const file = absolutePathParts(path);
  const library = absolutePathParts(root);
  if (!file || !library || file.volume !== library.volume) return null;
  const parent = library.parts.slice();
  if (parent[parent.length - 1] === "") parent.pop();
  const unsafe = (part: string) => part === "" || part === "." || part === ".."
    || (file.volume !== "/" && /[. ]$/.test(part));
  // Exact directory casing also works on NTFS directories configured as case-sensitive.
  if (file.parts.some(unsafe) || parent.some(unsafe) || file.parts.length <= parent.length
      || !parent.every((part, index) => part === file.parts[index])) {
    return null;
  }
  return file.parts.slice(parent.length).join(file.separator);
}

export function pathFilename(path: string) {
  const parts = absolutePathParts(path)?.parts ?? path.split(path.startsWith("\\\\") ? /[\\/]/ : "/");
  return parts[parts.length - 1] || path;
}
