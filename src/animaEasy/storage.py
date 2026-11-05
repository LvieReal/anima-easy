import json
import os
import pathlib
import tempfile

from animaEasy.errors import StorageError


SCHEMA_VERSION = 1


def readJson(path: pathlib.Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, UnicodeError, ValueError) as error:
        raise StorageError(f"Cannot read {path}: {error}. The file has not been replaced.") from error
    if not isinstance(data, dict) or type(data.get("schemaVersion")) is not int or data["schemaVersion"] != SCHEMA_VERSION:
        raise StorageError(f"Unsupported or missing schema version in {path}. The file has not been replaced.")
    return data


def writeJson(path: pathlib.Path, data: dict) -> None:
    temporaryPath = None
    try:
        content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporaryPath = pathlib.Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporaryPath, path)
    except (OSError, TypeError, ValueError) as error:
        raise StorageError(f"Cannot save {path}: {error}") from error
    finally:
        if temporaryPath is not None:
            temporaryPath.unlink(missing_ok=True)


def validateFolder(path: str, label: str = "workspace") -> pathlib.Path:
    if not path.strip():
        raise StorageError(f"Choose a {label} folder.")
    folder = pathlib.Path(path).expanduser().resolve()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=folder):
            pass
    except OSError as error:
        raise StorageError(f"The {label} is not writable: {folder}. {error}") from error
    return folder
