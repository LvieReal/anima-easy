import dataclasses
import os
import pathlib
import struct

import safetensors

from animaEasy.domain import MODEL_ROLES, newId, requireText, timestamp, validateId
from animaEasy.errors import StorageError, ValidationError
from animaEasy.storage import readJson, writeJson


@dataclasses.dataclass(frozen=True)
class ModelInspection:
    path: str
    sizeBytes: int
    modifiedNs: int
    tensorCount: int


@dataclasses.dataclass(frozen=True)
class ModelRecord:
    id: str
    label: str
    role: str
    variant: str
    path: str
    sizeBytes: int
    modifiedNs: int
    tensorCount: int
    registeredAt: str

    def validate(self) -> None:
        validateId(self.id)
        for name in ("label", "role", "variant", "path", "registeredAt"):
            requireText(getattr(self, name), name)
        if self.role not in MODEL_ROLES or not self.label.strip() or not self.path.strip():
            raise ValidationError("Model records need a label, a valid role, and a path.")
        if not pathlib.Path(self.path).is_absolute() or pathlib.Path(self.path).suffix.lower() != ".safetensors":
            raise ValidationError("Registered model paths must be absolute .safetensors paths.")
        if type(self.modifiedNs) is not int:
            raise ValidationError("Invalid model modification timestamp.")
        for name in ("sizeBytes", "tensorCount"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValidationError(f"Invalid model {name}.")

    def fileStatus(self) -> str:
        try:
            path = pathlib.Path(self.path)
            if not path.is_file():
                return "Missing file"
            stat = path.stat()
            if stat.st_size != self.sizeBytes or stat.st_mtime_ns != self.modifiedNs:
                return "Changed; register again"
        except OSError:
            return "File unavailable"
        return "Registered"


def inspectModel(path: str) -> ModelInspection:
    if not path.strip():
        raise ValidationError("Choose a .safetensors file.")
    modelPath = pathlib.Path(path).expanduser().resolve()
    if modelPath.suffix.lower() != ".safetensors":
        raise ValidationError("Model registration accepts .safetensors files only. Other checkpoint formats are not loaded.")
    try:
        before = modelPath.stat()
        if not modelPath.is_file():
            raise ValidationError("Choose a file, not a folder.")
        with modelPath.open("rb") as stream:
            sizeHeader = stream.read(8)
        if len(sizeHeader) != 8:
            raise ValidationError("The model is empty or truncated.")
        headerLength = struct.unpack("<Q", sizeHeader)[0]
        if not 2 <= headerLength <= 16 * 1024 * 1024 or headerLength + 8 > before.st_size:
            raise ValidationError("Invalid or oversized safetensors header. The file may be an incomplete download.")
        with safetensors.safe_open(modelPath, framework="numpy") as model:
            tensorCount = len(model.keys())
        if tensorCount == 0:
            raise ValidationError("The file does not contain any tensors.")
        after = modelPath.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValidationError("The file changed while it was being inspected. Wait for the copy or download to finish.")
        return ModelInspection(str(modelPath), after.st_size, after.st_mtime_ns, tensorCount)
    except (OSError, safetensors.SafetensorError) as error:
        raise ValidationError(f"Cannot inspect {modelPath.name}: {error}") from error


class ModelLibrary:
    def __init__(self, statePath: pathlib.Path):
        self.path = statePath / "models.json"
        self.records = self.readRecords()

    def readRecords(self) -> list[ModelRecord]:
        if not self.path.exists():
            return []
        data = readJson(self.path)
        if not isinstance(data.get("models"), list):
            raise StorageError(f"Invalid model library: {self.path}")
        try:
            records = [ModelRecord(**item) for item in data["models"]]
            for record in records:
                record.validate()
        except (TypeError, ValidationError) as error:
            raise StorageError(f"Invalid model library: {error}. The file has not been replaced.") from error
        if len({record.id for record in records}) != len(records):
            raise StorageError("The model library contains duplicate IDs. The file has not been replaced.")
        return records

    def saveRecords(self, records: list[ModelRecord]) -> None:
        writeJson(self.path, {"schemaVersion": 1, "models": [dataclasses.asdict(record) for record in records]})
        self.records = records

    def register(self, inspection: ModelInspection, role: str, label: str = "", variant: str = "") -> ModelRecord:
        if role not in MODEL_ROLES:
            raise ValidationError("Choose a model role.")
        canonicalPath = os.path.normcase(inspection.path)
        existing = next((record for record in self.records if os.path.normcase(record.path) == canonicalPath), None)
        if existing is not None and existing.role != role:
            raise ValidationError(f"This file is already registered as {MODEL_ROLES[existing.role]}. Forget that entry before changing its role.")
        record = ModelRecord(existing.id if existing else newId(), label.strip() or pathlib.Path(inspection.path).stem, role, variant, inspection.path, inspection.sizeBytes, inspection.modifiedNs, inspection.tensorCount, timestamp())
        record.validate()
        if record.fileStatus() != "Registered":
            raise ValidationError("The model changed or disappeared before registration. Inspect it again.")
        self.saveRecords([item for item in self.records if item.id != record.id] + [record])
        return record

    def forget(self, modelId: str) -> None:
        self.get(modelId)
        self.saveRecords([record for record in self.records if record.id != modelId])

    def get(self, modelId: str) -> ModelRecord:
        record = next((record for record in self.records if record.id == modelId), None)
        if record is None:
            raise ValidationError("A selected model is no longer in the library. Select another model.")
        return record

    def choices(self, role: str) -> list[ModelRecord]:
        return sorted((record for record in self.records if record.role == role), key=lambda record: record.label.casefold())
