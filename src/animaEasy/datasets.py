import dataclasses
import threading
import warnings

import PIL.Image

from animaEasy.captions import CaptionProfile, checkCaptions
from animaEasy.errors import ValidationError
from animaEasy.images import listImages


MAX_ISSUES = 100


@dataclasses.dataclass(frozen=True)
class DatasetReport:
    path: str
    imageCount: int
    captionCount: int
    missingCaptions: int
    invalidFiles: int
    nestedFolders: int
    issues: tuple[str, ...]
    captionMode: str = "matching"

    @property
    def usable(self) -> bool:
        return self.imageCount > 0 and self.invalidFiles == 0


def inspectDataset(path: str, cancelEvent: threading.Event | None = None, captionProfile: CaptionProfile | None = None) -> DatasetReport:
    index = listImages(path, cancelEvent)
    profile = captionProfile or CaptionProfile()
    captions = checkCaptions(index.folder, index.names, profile, cancelEvent)
    captionCount = 0
    missingCaptions = 0
    invalidFiles = 0
    issues = []

    def recordIssue(message: str) -> None:
        if len(issues) < MAX_ISSUES:
            issues.append(message)

    for name in index.names:
        if cancelEvent is not None and cancelEvent.is_set():
            raise ValidationError("Dataset check canceled.")
        imagePath = index.folder / name
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", PIL.Image.DecompressionBombWarning)
                with PIL.Image.open(imagePath) as image:
                    image.verify()
        except (OSError, ValueError, SyntaxError, PIL.Image.DecompressionBombError, PIL.Image.DecompressionBombWarning) as error:
            invalidFiles += 1
            recordIssue(f"Unreadable image: {name} ({error})")
        caption = captions[name]
        if caption.status == "Ready":
            captionCount += 1
        elif caption.status == "Invalid":
            invalidFiles += 1
            recordIssue(f"{name}: {caption.issue}")
        else:
            missingCaptions += 1
            recordIssue(f"{name}: {caption.issue}")
    if not index.names:
        recordIssue("No supported images found in the selected folder.")
    if index.nestedFolders:
        recordIssue(f"{index.nestedFolders} subfolder(s) ignored. Only the selected folder is checked.")
    return DatasetReport(str(index.folder), len(index.names), captionCount, missingCaptions, invalidFiles, index.nestedFolders, tuple(issues), profile.mode)
