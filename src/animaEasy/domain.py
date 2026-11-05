import dataclasses
import datetime
import math
import re
import uuid

from animaEasy.captions import CaptionProfile, datasetScope
from animaEasy.errors import ValidationError
from animaEasy.validation import requireText


MODEL_ROLES = {"diffusion": "Diffusion model", "textEncoder": "Text encoder", "vae": "VAE"}
BASE_MODELS = ["Anima base v1.0", "Anima Preview3 base", "Anima Preview", "Custom Anima model"]
EFFORT_PRESETS = {"Custom": None, "Low": (5, 5), "Medium": (10, 10), "High": (20, 10)}


def timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="microseconds")


def newId() -> str:
    return uuid.uuid4().hex


def validateId(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{32}", value) is None:
        raise ValidationError("Invalid record ID.")
    return value


@dataclasses.dataclass(frozen=True)
class TrainingSettings:
    rank: int = 20
    alpha: int = 20
    learningRate: float = 0.0001
    epochs: int = 10
    resolution: int = 768
    repeats: int = 10
    batchSize: int = 1
    gradientAccumulation: int = 1
    seed: int = 42
    mixedPrecision: str = "bf16"
    gradientCheckpointing: bool = True
    cacheLatents: bool = True
    cacheTextEncoder: bool = True
    effortPreset: str = "Custom"
    blocksToSwap: int = 0
    optimizer: str = "AdamW"
    vaeChunkSize: int = 128
    keepCheckpoints: int = 3

    def validate(self) -> None:
        if not isinstance(self.effortPreset, str) or self.effortPreset not in EFFORT_PRESETS:
            raise ValidationError("Choose Custom, Low, Medium, or High training effort.")
        limits = {"rank": (1, 512), "alpha": (1, 512), "epochs": (1, 10000), "resolution": (256, 2048), "repeats": (1, 10000), "batchSize": (1, 64), "gradientAccumulation": (1, 256), "seed": (0, 2147483647), "blocksToSwap": (0, 26), "vaeChunkSize": (0, 512), "keepCheckpoints": (1, 50)}
        for name, (minimum, maximum) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValidationError(f"{name} must be an integer between {minimum} and {maximum}.")
        if self.resolution % 64:
            raise ValidationError("Resolution must be a multiple of 64.")
        if self.vaeChunkSize % 2:
            raise ValidationError("VAE chunk size must be zero or an even number.")
        if self.optimizer not in ("AdamW", "AdamW8bit"):
            raise ValidationError("Choose AdamW or AdamW8bit.")
        if type(self.learningRate) not in (int, float) or not math.isfinite(self.learningRate) or not 0 < self.learningRate <= 1:
            raise ValidationError("Learning rate must be greater than zero and at most 1.")
        if self.mixedPrecision not in ("bf16", "fp16", "no"):
            raise ValidationError("Choose bf16, fp16, or no mixed precision.")
        for name in ("gradientCheckpointing", "cacheLatents", "cacheTextEncoder"):
            if type(getattr(self, name)) is not bool:
                raise ValidationError(f"{name} must be true or false.")

    def usePreset(self, name: str):
        if name not in EFFORT_PRESETS:
            raise ValidationError("Unknown training preset.")
        effort = EFFORT_PRESETS[name]
        settings = dataclasses.replace(self, effortPreset=name) if effort is None else dataclasses.replace(self, effortPreset=name, epochs=effort[0], repeats=effort[1])
        settings.validate()
        return settings

    @classmethod
    def fromData(cls, data: dict):
        if not isinstance(data, dict):
            raise ValidationError("Training settings must be an object.")
        try:
            settings = cls(**data)
        except TypeError as error:
            raise ValidationError(f"Invalid training settings: {error}") from error
        settings.validate()
        effort = EFFORT_PRESETS[settings.effortPreset]
        if effort is not None and (settings.epochs, settings.repeats) != effort:
            settings = dataclasses.replace(settings, effortPreset="Custom")
        return settings


@dataclasses.dataclass(frozen=True)
class Project:
    id: str
    name: str
    createdAt: str
    updatedAt: str
    datasetPath: str = ""
    notes: str = ""
    modelIds: dict[str, str] = dataclasses.field(default_factory=dict)
    training: TrainingSettings = dataclasses.field(default_factory=TrainingSettings)
    currentStep: int = 0
    captionProfiles: dict[str, CaptionProfile] = dataclasses.field(default_factory=dict)

    def validate(self) -> None:
        validateId(self.id)
        requireText(self.name, "Project name", 100)
        if not self.name.strip():
            raise ValidationError("Enter a project name.")
        requireText(self.datasetPath, "Dataset path")
        requireText(self.notes, "Project notes", 4000)
        for value in (self.createdAt, self.updatedAt):
            try:
                parsed = datetime.datetime.fromisoformat(requireText(value, "Timestamp", 80))
            except ValueError as error:
                raise ValidationError("Invalid project timestamp.") from error
            if parsed.tzinfo is None:
                raise ValidationError("Project timestamps must include a time zone.")
        if type(self.currentStep) is not int or not 0 <= self.currentStep <= 3:
            raise ValidationError("Invalid project step.")
        if not isinstance(self.modelIds, dict) or any(role not in MODEL_ROLES for role in self.modelIds):
            raise ValidationError("Invalid model roles in project.")
        for modelId in self.modelIds.values():
            validateId(modelId)
        if not isinstance(self.captionProfiles, dict):
            raise ValidationError("Dataset caption profiles must be an object.")
        for folder, profile in self.captionProfiles.items():
            requireText(folder, "Caption dataset folder")
            if not folder or not isinstance(profile, CaptionProfile):
                raise ValidationError("Invalid dataset caption profile.")
            profile.validate()
        self.training.validate()

    @property
    def captions(self) -> CaptionProfile:
        return self.captionProfiles.get(datasetScope(self.datasetPath), CaptionProfile())

    def withCaptions(self, profile: CaptionProfile):
        scope = datasetScope(self.datasetPath)
        if not scope:
            raise ValidationError("Choose a dataset folder before assigning captions.")
        profile.validate()
        return dataclasses.replace(self, captionProfiles={**self.captionProfiles, scope: profile})

    @classmethod
    def create(cls, name: str):
        createdAt = timestamp()
        project = cls(id=newId(), name=name.strip(), createdAt=createdAt, updatedAt=createdAt)
        project.validate()
        return project

    def toData(self) -> dict:
        return {"schemaVersion": 1, **dataclasses.asdict(self)}

    @classmethod
    def fromData(cls, data: dict):
        values = {name: value for name, value in data.items() if name != "schemaVersion"}
        values["training"] = TrainingSettings.fromData(values.get("training", {}))
        profiles = values.get("captionProfiles", {})
        if not isinstance(profiles, dict):
            raise ValidationError("Dataset caption profiles must be an object.")
        values["captionProfiles"] = {folder: CaptionProfile.fromData(profile) for folder, profile in profiles.items()}
        try:
            project = cls(**values)
        except TypeError as error:
            raise ValidationError(f"Invalid project: {error}") from error
        project.validate()
        return project
