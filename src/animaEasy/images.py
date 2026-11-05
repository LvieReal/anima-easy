import array
import dataclasses
import pathlib
import threading
import warnings

import PIL.Image
import PIL.ImageOps

from animaEasy.errors import ValidationError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclasses.dataclass(frozen=True)
class ImageIndex:
    folder: pathlib.Path
    names: tuple[str, ...]
    nestedFolders: int


@dataclasses.dataclass(frozen=True)
class ImagePreview:
    width: int
    height: int
    originalSize: tuple[int, int]
    pixels: array.array


def listImages(path: str, cancelEvent: threading.Event | None = None) -> ImageIndex:
    if not path.strip():
        raise ValidationError("Choose a dataset folder first.")
    folder = pathlib.Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise ValidationError(f"Dataset folder not found: {folder}")
    names = []
    nestedFolders = 0
    try:
        for entry in folder.iterdir():
            if cancelEvent is not None and cancelEvent.is_set():
                raise ValidationError("Image indexing canceled.")
            if entry.is_dir():
                nestedFolders += 1
            elif entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
                names.append(entry.name)
    except OSError as error:
        raise ValidationError(f"Cannot list dataset images: {error}") from error
    return ImageIndex(folder, tuple(sorted(names, key=str.casefold)), nestedFolders)


def loadPreview(path: pathlib.Path) -> ImagePreview:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PIL.Image.DecompressionBombWarning)
            with PIL.Image.open(path) as image:
                originalSize = image.size
                image.draft("RGB", (192, 144))
                image = PIL.ImageOps.exif_transpose(image)
                image.thumbnail((192, 144), PIL.Image.Resampling.LANCZOS)
                preview = image.convert("RGBA")
                pixels = array.array("f", (channel / 255 for channel in preview.tobytes()))
                return ImagePreview(preview.width, preview.height, originalSize, pixels)
    except (OSError, ValueError, SyntaxError, PIL.Image.DecompressionBombError, PIL.Image.DecompressionBombWarning) as error:
        raise ValidationError(f"Cannot preview {path.name}: {error}") from error
