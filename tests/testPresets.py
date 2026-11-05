import dataclasses

import pytest

from animaEasy.domain import EFFORT_PRESETS, TrainingSettings
from animaEasy.errors import ValidationError


@pytest.mark.parametrize("name, expected", [("Low", (5, 5)), ("Medium", (10, 10)), ("High", (20, 10))])
def testEffortPresetsOnlyChangeEpochsRepeatsAndThePresetLabel(name, expected):
    settings = TrainingSettings(rank=32, alpha=16, resolution=512, learningRate=0.0002, batchSize=2, gradientAccumulation=3, seed=123, mixedPrecision="fp16", gradientCheckpointing=False, cacheLatents=False, cacheTextEncoder=False)
    updated = settings.usePreset(name)
    assert (updated.epochs, updated.repeats) == expected
    assert updated.effortPreset == name
    before, after = dataclasses.asdict(settings), dataclasses.asdict(updated)
    for field in before:
        if field not in ("epochs", "repeats", "effortPreset"):
            assert after[field] == before[field]
    assert TrainingSettings.fromData(after) == updated


def testCustomDoesNotResetExistingSettings():
    high = TrainingSettings(rank=64).usePreset("High")
    custom = high.usePreset("Custom")
    assert custom == dataclasses.replace(high, effortPreset="Custom")
    assert EFFORT_PRESETS["Custom"] is None


def testMismatchedStoredPresetBecomesCustomWithoutChangingCounts():
    data = dataclasses.asdict(TrainingSettings().usePreset("Low"))
    data["epochs"] = 7
    restored = TrainingSettings.fromData(data)
    assert restored.effortPreset == "Custom"
    assert (restored.epochs, restored.repeats) == (7, 5)


def testUnknownPresetIsRejected():
    with pytest.raises(ValidationError):
        TrainingSettings().usePreset("Ultra")
    with pytest.raises(ValidationError):
        TrainingSettings.fromData({"effortPreset": "Ultra"})
