import hashlib
import hmac
import json
import os
import pathlib
import sys
import tempfile


EVENT_PREFIX = "ANIMA_EASY_EVENT "
STATE_MARKER = "animaEasyState.json"


def readData(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def writeData(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporaryPath = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".animaEasy-", suffix=".tmp", delete=False) as stream:
            temporaryPath = pathlib.Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporaryPath, path)
    finally:
        if temporaryPath is not None:
            temporaryPath.unlink(missing_ok=True)


def fileDigest(path: pathlib.Path, cancel=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Canceled while checking files.")
            digest.update(chunk)
    return digest.hexdigest()


def captureFile(path: pathlib.Path, cancel=None) -> dict:
    path = path.resolve(strict=True)
    before = path.stat()
    digest = fileDigest(path, cancel)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"File changed while being read: {path}")
    return {"path": str(path), "sizeBytes": after.st_size, "modifiedNs": after.st_mtime_ns, "sha256": digest}


def verifyFile(reference: dict, cancel=None) -> None:
    path = pathlib.Path(reference["path"])
    if not path.is_file() or path.stat().st_size != reference["sizeBytes"] or fileDigest(path, cancel) != reference["sha256"]:
        raise ValueError(f"A captured input is missing or changed: {path}. Restore the original file or start a new run.")


def cacheStem(folder: pathlib.Path, imagePath: str) -> pathlib.Path:
    return folder / hashlib.sha256(imagePath.encode("utf-8")).hexdigest()[:32]


def isLink(path: pathlib.Path) -> bool:
    return path.is_symlink() or path.is_junction()


def stateFiles(folder: pathlib.Path) -> list[pathlib.Path]:
    if isLink(folder):
        raise ValueError("Trainer state folders cannot be symbolic links.")
    files = []
    for path in sorted(folder.rglob("*")):
        if isLink(path):
            raise ValueError("Trainer states cannot contain symbolic links.")
        if path.is_file() and path.name != STATE_MARKER:
            files.append(path)
    return files


def sealState(folder: pathlib.Path, runId: str, recipeHash: str, completedEpochs: int, completedSteps: int, stateKey: bytes) -> dict:
    files = stateFiles(folder)
    names = {path.name for path in files}
    if not {"optimizer.bin", "scheduler.bin", "random_states_0.pkl"}.issubset(names) or not any(name.startswith("model") and name.endswith(".safetensors") for name in names):
        raise ValueError("Trainer state is missing model, optimizer, scheduler, or random-generator state.")
    if completedEpochs < 1 or completedSteps < 1:
        raise ValueError("An empty trainer state is not resumable.")
    data = {"schemaVersion": 1, "runId": runId, "recipeHash": recipeHash, "completedEpochs": completedEpochs, "completedSteps": completedSteps, "files": [{"name": str(path.relative_to(folder)), "sizeBytes": path.stat().st_size, "sha256": fileDigest(path)} for path in files]}
    data["signature"] = stateSignature(data, stateKey)
    writeData(folder / STATE_MARKER, data)
    return data


def stateSignature(data: dict, stateKey: bytes) -> str:
    if len(stateKey) != 32:
        raise ValueError("The local trainer-state ownership key is invalid.")
    content = json.dumps({name: value for name, value in data.items() if name != "signature"}, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hmac.new(stateKey, content, "sha256").hexdigest()


def readStateMarker(folder: pathlib.Path, stateKey: bytes) -> dict:
    markerPath = folder / STATE_MARKER
    if isLink(folder) or isLink(markerPath) or markerPath.stat().st_size > 1024 * 1024:
        raise ValueError("Invalid trainer state marker.")
    data = readData(markerPath)
    signature = data.get("signature")
    if not isinstance(signature, str) or not hmac.compare_digest(signature, stateSignature(data, stateKey)):
        raise ValueError("This trainer state is not authenticated by this application profile. Imported or modified states cannot be resumed.")
    return data


def verifyState(folder: pathlib.Path, runId: str, recipeHash: str, stateKey: bytes, cancel=None) -> dict:
    data = readStateMarker(folder, stateKey)
    if data.get("schemaVersion") != 1 or data.get("runId") != runId or data.get("recipeHash") != recipeHash:
        raise ValueError("This state does not belong to the captured run configuration.")
    if type(data.get("completedEpochs")) is not int or data["completedEpochs"] < 1 or type(data.get("completedSteps")) is not int or data["completedSteps"] < 1:
        raise ValueError("Invalid completed-epoch state.")
    files = stateFiles(folder)
    recorded = data.get("files")
    if not isinstance(recorded, list) or {str(path.relative_to(folder)) for path in files} != {item["name"] for item in recorded} or len(files) != len(recorded):
        raise ValueError("The trainer state is incomplete or contains unexpected files.")
    for item in recorded:
        relativePath = pathlib.PurePath(item["name"])
        if relativePath.is_absolute() or ".." in relativePath.parts or "\\" in item["name"] and os.name != "nt":
            raise ValueError("Invalid trainer state filename.")
        path = folder / relativePath
        if path.stat().st_size != item["sizeBytes"] or fileDigest(path, cancel) != item["sha256"]:
            raise ValueError(f"Trainer state file changed or is incomplete: {path.name}")
    return data


def emitEvent(token: str, kind: str, **values) -> None:
    content = EVENT_PREFIX + json.dumps({"token": token, "kind": kind, **values}, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n" + content + "\n")
    sys.stdout.flush()
