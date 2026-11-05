import dataclasses

import pytest

from animaEasy.domain import Project, TrainingSettings
from animaEasy.errors import ValidationError


def testProjectRoundTrip():
    project = Project.create("  Étude — Модель  ")
    restored = Project.fromData(project.toData())
    assert restored == project
    assert restored.name == "Étude — Модель"


@pytest.mark.parametrize("name", ["", "   ", "a" * 101, "bad\x00name"])
def testInvalidProjectNames(name):
    with pytest.raises(ValidationError):
        Project.create(name)


@pytest.mark.parametrize("changes", [{"rank": 0}, {"rank": True}, {"alpha": -1}, {"epochs": 1.5}, {"resolution": 777}, {"learningRate": float("nan")}, {"learningRate": float("inf")}, {"learningRate": 0}, {"mixedPrecision": "unknown"}, {"cacheLatents": "true"}, {"seed": -1}])
def testInvalidTrainingSettings(changes):
    settings = dataclasses.replace(TrainingSettings(), **changes)
    with pytest.raises(ValidationError):
        settings.validate()


def testSettingsRoundTrip():
    settings = dataclasses.replace(TrainingSettings(), rank=32, alpha=16, mixedPrecision="fp16")
    assert TrainingSettings.fromData(dataclasses.asdict(settings)) == settings


def testUntrustedProjectFields():
    data = Project.create("Example").toData()
    data["modelIds"] = {"diffusion": "../../outside"}
    with pytest.raises(ValidationError):
        Project.fromData(data)
