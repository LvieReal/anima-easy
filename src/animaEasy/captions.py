import dataclasses
import os
import pathlib
import threading
from collections.abc import Iterable

from animaEasy.errors import ValidationError
from animaEasy.validation import requireText


CAPTION_MODES = {"matching": "Matching .txt files", "files": "Assigned caption files", "inline": "Text inside the project"}
MAX_CAPTION_LENGTH = 16384
MAX_CAPTION_FILE_BYTES = 1024 * 1024


def datasetScope(path: str) -> str:
    if not path.strip():
        return ""
    return os.path.normcase(os.path.abspath(os.path.expanduser(path)))


def validateImageName(name: str) -> None:
    requireText(name, "Image filename")
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValidationError("Caption assignments must use a filename, not a path.")


@dataclasses.dataclass(frozen=True)
class CaptionProfile:
    mode: str = "matching"
    fileAssignments: dict[str, str] = dataclasses.field(default_factory=dict)
    textCaptions: dict[str, str] = dataclasses.field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in CAPTION_MODES:
            raise ValidationError("Choose a supported caption source.")
        for mapping, limit in ((self.fileAssignments, 4096), (self.textCaptions, MAX_CAPTION_LENGTH)):
            if not isinstance(mapping, dict):
                raise ValidationError("Caption assignments must be an object.")
            for name, value in mapping.items():
                validateImageName(name)
                requireText(value, "Caption value", limit)

    @classmethod
    def fromData(cls, data: dict):
        if not isinstance(data, dict):
            raise ValidationError("Caption settings must be an object.")
        try:
            profile = cls(**data)
        except TypeError as error:
            raise ValidationError(f"Invalid caption settings: {error}") from error
        profile.validate()
        return profile

    def assignFile(self, names: Iterable[str], path: str):
        requireText(path, "Caption path")
        assignments = dict(self.fileAssignments)
        for name in names:
            validateImageName(name)
            if path.strip():
                assignments[name] = path
            else:
                assignments.pop(name, None)
        return dataclasses.replace(self, fileAssignments=assignments)

    def assignText(self, names: Iterable[str], text: str):
        requireText(text, "Caption", MAX_CAPTION_LENGTH)
        captions = dict(self.textCaptions)
        for name in names:
            validateImageName(name)
            if text:
                captions[name] = text
            else:
                captions.pop(name, None)
        return dataclasses.replace(self, textCaptions=captions)


@dataclasses.dataclass(frozen=True)
class CaptionDocument:
    text: str
    source: str
    status: str
    issue: str = ""


@dataclasses.dataclass(frozen=True)
class CaptionCheck:
    status: str
    issue: str = ""


def readCaptionFile(path: str) -> CaptionDocument:
    if not path.strip():
        return CaptionDocument("", "", "Unassigned", "No caption file assigned.")
    try:
        captionPath = pathlib.Path(path).expanduser().resolve()
        if not captionPath.is_file():
            return CaptionDocument("", str(captionPath), "Missing", f"Caption file not found: {captionPath}")
        with captionPath.open("rb") as stream:
            content = stream.read(MAX_CAPTION_FILE_BYTES + 1)
        if len(content) > MAX_CAPTION_FILE_BYTES:
            return CaptionDocument("", str(captionPath), "Invalid", f"Caption exceeds 1 MiB: {captionPath.name}")
        text = content.decode("utf-8-sig")
        if "\x00" in text:
            return CaptionDocument("", str(captionPath), "Invalid", f"Caption contains null characters: {captionPath.name}")
        status = "Ready" if text.strip() else "Empty"
        return CaptionDocument(text, str(captionPath), status, "" if status == "Ready" else f"Empty caption: {captionPath.name}")
    except (OSError, UnicodeError, ValueError) as error:
        return CaptionDocument("", path, "Invalid", f"Cannot read caption: {error}")


def captionSource(folder: pathlib.Path, imageName: str, profile: CaptionProfile) -> str:
    validateImageName(imageName)
    if profile.mode == "matching":
        return str((folder / imageName).with_suffix(".txt"))
    if profile.mode == "files":
        value = profile.fileAssignments.get(imageName, "")
        if not value.strip():
            return ""
        path = pathlib.Path(value).expanduser()
        return str(path if path.is_absolute() else folder / path)
    return "Project text"


def resolveCaption(folder: pathlib.Path, imageName: str, profile: CaptionProfile) -> CaptionDocument:
    profile.validate()
    validateImageName(imageName)
    if profile.mode == "inline":
        text = profile.textCaptions.get(imageName, "")
        return CaptionDocument(text, "Project text", "Ready" if text.strip() else "Empty", "" if text.strip() else f"No project caption for {imageName}.")
    return readCaptionFile(captionSource(folder, imageName, profile))


def checkCaptions(folder: pathlib.Path, names: Iterable[str], profile: CaptionProfile, cancelEvent: threading.Event | None = None) -> dict[str, CaptionCheck]:
    profile.validate()
    checkedFiles = {}
    results = {}
    for name in names:
        if cancelEvent is not None and cancelEvent.is_set():
            raise ValidationError("Caption check canceled.")
        if profile.mode == "inline":
            text = profile.textCaptions.get(name, "")
            results[name] = CaptionCheck("Ready" if text.strip() else "Empty", "" if text.strip() else "No project caption.")
            continue
        source = captionSource(folder, name, profile)
        if source not in checkedFiles:
            document = readCaptionFile(source)
            checkedFiles[source] = CaptionCheck(document.status, document.issue)
        results[name] = checkedFiles[source]
    return results
