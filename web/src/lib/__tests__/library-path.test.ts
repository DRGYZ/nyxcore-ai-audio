import { describe, expect, it } from "vitest";
import { libraryRelativePath, pathFilename } from "../library-path";

describe("library path presentation", () => {
  it("shortens Windows paths with mixed separators without losing Unicode or long directory names", () => {
    const album = "非常に長い".repeat(20);
    const path = "c:/Music Library\\" + album + "/Café 録音.flac";
    expect(libraryRelativePath(path, "C:\\Music Library\\")).toBe(album + "\\Café 録音.flac");
    expect(pathFilename(path)).toBe("Café 録音.flac");
  });

  it("uses exact component boundaries and preserves POSIX backslashes", () => {
    expect(libraryRelativePath("/home/user/music/Été\\live/曲.mp3", "/home/user/music/")).toBe("Été\\live/曲.mp3");
    expect(libraryRelativePath("/home/user/music-backup/曲.mp3", "/home/user/music")).toBeNull();
    expect(libraryRelativePath("/home/user/music/../other/曲.mp3", "/home/user/music")).toBeNull();
    expect(libraryRelativePath("/home/user/Music/曲.mp3", "/home/user/music")).toBeNull();
    expect(libraryRelativePath("D:\\Music Library\\曲.mp3", "C:\\Music Library")).toBeNull();
    expect(libraryRelativePath("C:\\Music Library\\.. \\曲.mp3", "C:\\Music Library")).toBeNull();
    expect(libraryRelativePath("C:曲.mp3", "C:\\Music Library")).toBeNull();
    expect(libraryRelativePath("\\\\server\\share\\曲.mp3", "\\\\server\\share")).toBeNull();
  });
});
