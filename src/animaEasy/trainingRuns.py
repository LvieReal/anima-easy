import dataclasses
import pathlib
import shutil
import threading
import warnings

import PIL.Image
import psutil

from animaEasy.captions import captionSource, readCaptionFile
from animaEasy.domain import MODEL_ROLES, Project, newId, timestamp, validateId
from animaEasy.errors import AnimaError, ValidationError
from animaEasy.images import listImages
from animaEasy.storage import readJson, writeJson
from animaEasy.trainingAssets import runtimeContract
from animaEasy.trainingRuntime import ASSET_FOLDER, TRAINER_REVISION


ACTIVE_STATUSES = {"preparing", "running", "stopping"}


def processAlive(processId, createdAt) -> bool:
    if type(processId) is not int or not isinstance(createdAt, (int, float)):
        return False
    try:
        process = psutil.Process(processId)
        return abs(process.create_time() - createdAt) < 0.01 and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def captureInputs(folder: pathlib.Path, project: Project, models: dict, runtimeFingerprint: str, cancel: threading.Event, onLine) -> dict:
    project.validate()
    inputs = folder / "inputs"
    inputs.mkdir(exist_ok=True)
    index = listImages(project.datasetPath, cancel)
    if not index.names:
        raise ValidationError("The selected dataset does not contain supported images.")
    modelData = {}
    for role, label in MODEL_ROLES.items():
        if role not in models or models[role].role != role:
            raise ValidationError(f"Choose a registered {label.lower()} before training.")
        record = models[role]
        if record.fileStatus() != "Registered":
            raise ValidationError(f"{label}: {record.fileStatus()}. Register the current file before training.")
        onLine(f"Checking existing {label.lower()}: {record.path}")
        modelData[role] = {"record": dataclasses.asdict(record), "file": runtimeContract.captureFile(pathlib.Path(record.path), cancel)}
    metadata = {}
    images = []
    captionFiles = {}
    emptyCount = 0
    for position, name in enumerate(index.names, 1):
        if cancel.is_set():
            raise InterruptedError("Input capture canceled.")
        imagePath = index.folder / name
        reference = runtimeContract.captureFile(imagePath, cancel)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PIL.Image.DecompressionBombWarning)
            with PIL.Image.open(imagePath) as image:
                imageSize = list(image.size)
                image.verify()
        if imagePath.resolve() != pathlib.Path(reference["path"]) or imagePath.stat().st_mtime_ns != reference["modifiedNs"] or imagePath.stat().st_size != reference["sizeBytes"]:
            raise ValidationError(f"Image changed while being checked: {name}")
        source = captionSource(index.folder, name, project.captions)
        if project.captions.mode == "inline":
            caption = project.captions.textCaptions.get(name, "")
            captionStatus = "Ready" if caption.strip() else "Empty"
        else:
            if source not in captionFiles:
                captionFiles[source] = readCaptionFile(source)
            document = captionFiles[source]
            if document.status == "Invalid":
                raise ValidationError(f"{name}: {document.issue}")
            caption, captionStatus = document.text, document.status
        if not caption.strip():
            emptyCount += 1
        metadata[str(imagePath)] = {"caption": caption, "image_size": imageSize}
        images.append({**reference, "datasetPath": str(imagePath), "imageSize": imageSize, "captionSource": source, "captionStatus": captionStatus})
        if position == 1 or position % 25 == 0 or position == len(index.names):
            onLine(f"Captured images and captions: {position}/{len(index.names)}")
    if emptyCount:
        onLine(f"Warning: {emptyCount} image(s) have empty or missing captions and will train with empty text.")
    runtimeContract.writeData(inputs / "captions.json", metadata)
    recipe = {"schemaVersion": 1, "runtimeFingerprint": runtimeFingerprint, "trainerRevision": TRAINER_REVISION, "project": project.toData(), "models": modelData, "images": images, "captionHash": runtimeContract.fileDigest(inputs / "captions.json"), "emptyCaptions": emptyCount}
    writeJson(inputs / "recipe.json", recipe)
    return recipe


def verifyInputs(folder: pathlib.Path, recipeHash: str, runtimeFingerprint: str, cancel: threading.Event, onLine) -> dict:
    inputs = folder / "inputs"
    if runtimeContract.fileDigest(inputs / "recipe.json", cancel) != recipeHash:
        raise ValidationError("The saved run configuration changed. It cannot be resumed safely.")
    recipe = readJson(inputs / "recipe.json")
    if recipe.get("runtimeFingerprint") != runtimeFingerprint:
        raise ValidationError("This run needs a different training runtime version. Its saved inputs have not been changed.")
    Project.fromData(recipe["project"])
    if runtimeContract.fileDigest(inputs / "captions.json", cancel) != recipe["captionHash"]:
        raise ValidationError("The captured captions changed. Restore the saved snapshot before resuming.")
    references = [data["file"] for data in recipe["models"].values()] + recipe["images"]
    for index, reference in enumerate(references, 1):
        if index <= 3 or index % 25 == 0 or index == len(references):
            onLine(f"Verifying original inputs: {index}/{len(references)}")
        runtimeContract.verifyFile(reference, cancel)
        if "datasetPath" in reference and pathlib.Path(reference["datasetPath"]).resolve(strict=True) != pathlib.Path(reference["path"]):
            raise ValidationError(f"A dataset image link changed: {reference['datasetPath']}")
    return recipe


class RunStore:
    def __init__(self, projects, stateKeyPath: pathlib.Path | None = None):
        self.projects = projects
        self.stateKeyPath = stateKeyPath

    def folder(self, projectId: str, runId: str) -> pathlib.Path:
        root = self.projects.folder(projectId) / "runs"
        folder = root / validateId(runId)
        if runtimeContract.isLink(root) or runtimeContract.isLink(folder) or any(runtimeContract.isLink(folder / name) for name in ("inputs", "outputs", "cache")):
            raise ValidationError("Training run directories cannot be symbolic links.")
        return folder

    def create(self, project: Project, runtimeFingerprint: str, resumeFrom: dict | None = None) -> dict:
        process = psutil.Process()
        record = {"schemaVersion": 1, "id": newId(), "projectId": project.id, "projectName": project.name, "createdAt": timestamp(), "updatedAt": timestamp(), "status": "preparing", "message": "Preparing captured inputs", "runtimeFingerprint": runtimeFingerprint, "trainerRevision": TRAINER_REVISION, "recipeHash": None, "resumeFrom": resumeFrom, "training": dataclasses.asdict(project.training), "completedSteps": 0, "totalSteps": None, "completedEpochs": 0, "ownerPid": process.pid, "ownerCreatedAt": process.create_time(), "processId": None, "processCreatedAt": None}
        folder = self.folder(project.id, record["id"])
        folder.mkdir(parents=True, exist_ok=False)
        for name in ("inputs", "outputs", "cache"):
            (folder / name).mkdir()
        self.save(record)
        return record

    def save(self, record: dict) -> None:
        record["updatedAt"] = timestamp()
        writeJson(self.folder(record["projectId"], record["id"]) / "run.json", record)

    def load(self, projectId: str, runId: str) -> dict:
        record = readJson(self.folder(projectId, runId) / "run.json")
        if record.get("id") != runId or record.get("projectId") != projectId or record.get("status") not in ACTIVE_STATUSES | {"completed", "canceled", "failed", "interrupted"}:
            raise ValidationError("The run record is invalid. It has not been replaced.")
        if record["status"] in ACTIVE_STATUSES and not processAlive(record.get("ownerPid"), record.get("ownerCreatedAt")):
            record.update({"status": "interrupted", "message": "The application exited before this run finished. Resume uses a completed saved epoch; unsaved work is lost."})
            self.save(record)
        return record

    def history(self, projectId: str | None = None) -> tuple[list[dict], list[str]]:
        folders = [self.projects.folder(projectId)] if projectId else list(self.projects.root.glob("*"))
        records, errors = [], []
        for projectFolder in folders:
            if not projectFolder.is_dir() or runtimeContract.isLink(projectFolder):
                continue
            for recordPath in (projectFolder / "runs").glob("*/run.json"):
                try:
                    records.append(self.load(projectFolder.name, recordPath.parent.name))
                except Exception as error:
                    errors.append(f"{recordPath}: {error}")
        return sorted(records, key=lambda item: item["createdAt"], reverse=True), errors

    def states(self, record: dict) -> list[dict]:
        folder = self.folder(record["projectId"], record["id"])
        result = []
        if self.stateKeyPath is None or not self.stateKeyPath.is_file():
            return result
        stateKey = self.stateKeyPath.read_bytes()
        for path in (folder / "outputs").glob("*-state"):
            try:
                if runtimeContract.isLink(path) or runtimeContract.isLink(path / runtimeContract.STATE_MARKER):
                    continue
                data = runtimeContract.readStateMarker(path, stateKey)
                if data.get("runId") == record["id"] and data.get("recipeHash") == record.get("recipeHash") and type(data.get("completedEpochs")) is int and type(data.get("completedSteps")) is int:
                    result.append({"name": path.name, "completedEpochs": data["completedEpochs"], "completedSteps": data["completedSteps"]})
            except (AnimaError, OSError, ValueError, TypeError, KeyError):
                continue
        return sorted(result, key=lambda item: item["completedSteps"], reverse=True)

    def checkpoints(self, record: dict) -> list[str]:
        output = self.folder(record["projectId"], record["id"]) / "outputs"
        result = []
        for marker in output.glob("*.safetensors.saved.json"):
            try:
                data = readJson(marker)
                path = output / marker.name.removesuffix(".saved.json")
                if not runtimeContract.isLink(path) and data.get("runId") == record["id"] and path.stat().st_size == data["sizeBytes"]:
                    result.append(path.name)
            except Exception:
                continue
        return sorted(result)

    def copySnapshot(self, source: pathlib.Path, destination: pathlib.Path) -> None:
        for name in ("recipe.json", "captions.json"):
            shutil.copyfile(source / "inputs" / name, destination / "inputs" / name)

    def copyBridge(self, folder: pathlib.Path) -> None:
        for name in ("runtimeBridge.py", "runtimeContract.py"):
            shutil.copyfile(ASSET_FOLDER / name, folder / "inputs" / name)
