import dataclasses
import os
import pathlib
import secrets
import threading
import tomllib

import pytest

from animaEasy.captions import CaptionProfile
from animaEasy.domain import Project, TrainingSettings
from animaEasy.errors import ValidationError
from animaEasy.storage import readJson
from animaEasy.trainingArguments import trainingArguments
from animaEasy.trainingAssets import runtimeContract
from animaEasy.trainingRuns import captureInputs, verifyInputs


def captureFixture(trainingInputs):
    workspace, project, models, controller, store, record, folder = trainingInputs
    recipe = captureInputs(folder, project, models, controller.runtime.fingerprint, threading.Event(), lambda message: None)
    record.update(recipeHash=runtimeContract.fileDigest(folder / "inputs" / "recipe.json"), imageCount=len(recipe["images"]), totalEpochs=project.training.epochs)
    store.save(record)
    return recipe


def testCapturesCaptionsWithoutCopyingOrRewritingInputs(trainingInputs):
    workspace, project, models, controller, store, record, folder = trainingInputs
    paths = list(pathlib.Path(project.datasetPath).iterdir()) + [pathlib.Path(model.path) for model in models.values()]
    originals = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    recipe = captureFixture(trainingInputs)
    captions = runtimeContract.readData(folder / "inputs" / "captions.json")
    assert len(recipe["images"]) == 3
    assert captions[str(pathlib.Path(project.datasetPath) / "image0.png")] == {"caption": "subject, example 0, portrait", "image_size": [64, 96]}
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == original for path, original in originals.items())
    assert not list(folder.rglob("*.safetensors"))
    assert not list(folder.rglob("*.png"))
    assert not list(folder.rglob("*.npz"))
    assert {data["file"]["path"] for data in recipe["models"].values()} == {model.path for model in models.values()}


@pytest.mark.parametrize("mode", ["inline", "files"])
def testCaptionSourcesAreCapturedAndLaterEditsDoNotAffectResume(trainingInputs, mode):
    workspace, project, models, controller, store, record, folder = trainingInputs
    shared = pathlib.Path(project.datasetPath) / "shared café.txt"
    shared.write_text("captured caption\r\ncafé", encoding="utf-8-sig", newline="")
    names = [f"image{index}.png" for index in range(3)]
    profile = CaptionProfile(mode=mode).assignFile(names, shared.name).assignText(names, "captured project text\nsecond line")
    project = project.withCaptions(profile)
    recipe = captureInputs(folder, project, models, controller.runtime.fingerprint, threading.Event(), lambda message: None)
    recipeHash = runtimeContract.fileDigest(folder / "inputs" / "recipe.json")
    shared.write_text("later external edit", encoding="utf-8")
    edited = dataclasses.replace(project.withCaptions(profile.assignText(names, "later project edit")), training=TrainingSettings(rank=64, epochs=99))
    workspace.projects.save(edited)
    restored = verifyInputs(folder, recipeHash, controller.runtime.fingerprint, threading.Event(), lambda message: None)
    assert restored == recipe
    assert restored["project"]["training"]["rank"] == 20
    captured = runtimeContract.readData(folder / "inputs" / "captions.json")
    expected = "captured project text\nsecond line" if mode == "inline" else "captured caption\r\ncafé"
    assert all(item["caption"] == expected for item in captured.values())
    assert shared.read_text(encoding="utf-8") == "later external edit"


@pytest.mark.parametrize("inputKind", ["image", "model"])
def testResumeChecksContentEvenWhenSizeAndTimestampArePreserved(trainingInputs, inputKind):
    workspace, project, models, controller, store, record, folder = trainingInputs
    captureFixture(trainingInputs)
    path = pathlib.Path(project.datasetPath) / "image0.png" if inputKind == "image" else pathlib.Path(models["diffusion"].path)
    previous = path.stat()
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    with pytest.raises(ValueError, match="missing or changed"):
        verifyInputs(folder, record["recipeHash"], controller.runtime.fingerprint, threading.Event(), lambda message: None)


def testResumeRejectsEditedSnapshotAndDifferentRuntime(trainingInputs):
    workspace, project, models, controller, store, record, folder = trainingInputs
    captureFixture(trainingInputs)
    with pytest.raises(ValidationError, match="different training runtime"):
        verifyInputs(folder, record["recipeHash"], "different runtime", threading.Event(), lambda message: None)
    path = folder / "inputs" / "recipe.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValidationError, match="configuration changed"):
        verifyInputs(folder, record["recipeHash"], controller.runtime.fingerprint, threading.Event(), lambda message: None)


def testArgumentGenerationProducesReadableTomlAndResumeBudget(trainingInputs):
    workspace, project, models, controller, store, record, folder = trainingInputs
    recipe = captureFixture(trainingInputs)
    arguments = trainingArguments(folder, recipe)
    config = tomllib.loads((folder / "inputs" / "dataset.toml").read_text(encoding="utf-8"))
    assert config["datasets"][0]["subsets"][0]["metadata_file"] == str(folder / "inputs" / "captions.json")
    assert config["datasets"][0]["batch_size"] == 1
    assert config["datasets"][0]["subsets"][0]["num_repeats"] == 2
    assert arguments[arguments.index("--network_module") + 1] == "networks.lora_anima"
    assert arguments[arguments.index("--max_train_epochs") + 1] == "3"
    resume = {"completedEpochs": 1, "path": str(folder / "outputs" / "anima-000001-state")}
    resumed = trainingArguments(folder, recipe, resume)
    assert resumed[resumed.index("--max_train_epochs") + 1] == "2"
    assert resumed[resumed.index("--initial_step") + 1] == "0"
    with pytest.raises(ValidationError, match="epoch target"):
        trainingArguments(folder, recipe, {**resume, "completedEpochs": 3})


def testMissingCaptionsAreExplicitlyCapturedAsEmpty(trainingInputs):
    workspace, project, models, controller, store, record, folder = trainingInputs
    (pathlib.Path(project.datasetPath) / "image0.txt").unlink()
    recipe = captureFixture(trainingInputs)
    assert recipe["emptyCaptions"] == 1
    assert runtimeContract.readData(folder / "inputs" / "captions.json")[str(pathlib.Path(project.datasetPath) / "image0.png")]["caption"] == ""


def testUnreadableCaptionsAndCancellationBlockPreparation(trainingInputs):
    workspace, project, models, controller, store, record, folder = trainingInputs
    canceled = threading.Event()
    canceled.set()
    with pytest.raises((ValidationError, InterruptedError), match="canceled|Canceled"):
        captureInputs(folder, project, models, controller.runtime.fingerprint, canceled, lambda message: None)
    (pathlib.Path(project.datasetPath) / "image0.txt").write_bytes(b"\xff\x00")
    with pytest.raises(ValidationError, match="caption"):
        captureInputs(folder, project, models, controller.runtime.fingerprint, threading.Event(), lambda message: None)


def testInterruptedHistoryPreservesRecordsAndIncompleteStates(trainingInputs):
    workspace, project, models, controller, store, record, folder = trainingInputs
    record.update(status="running", ownerCreatedAt=record["ownerCreatedAt"] - 50)
    store.save(record)
    incomplete = folder / "outputs" / "anima-000001-state"
    incomplete.mkdir()
    (incomplete / "optimizer.bin").write_bytes(b"incomplete")
    restored = store.load(project.id, record["id"])
    assert restored["status"] == "interrupted"
    assert store.states(restored) == []
    assert (incomplete / "optimizer.bin").read_bytes() == b"incomplete"
    assert store.history(project.id)[0][0]["id"] == record["id"]
    (folder / "run.json").write_text("damaged", encoding="utf-8")
    records, errors = store.history(project.id)
    assert not records and len(errors) == 1
    assert (folder / "run.json").read_text(encoding="utf-8") == "damaged"


def testStateOwnershipKeyIsPrivateStableAndNotSilentlyReplaced(trainingInputs):
    workspace, project, models, controller, store, record, folder = trainingInputs
    original = controller.stateKeyPath.read_bytes()
    controller.ensureStateKey()
    assert controller.stateKeyPath.read_bytes() == original
    if os.name == "posix":
        assert controller.stateKeyPath.stat().st_mode & 0o077 == 0
    controller.stateKeyPath.write_bytes(secrets.token_bytes(32))
    with pytest.raises(ValidationError, match="key changed"):
        controller.ensureStateKey()
    controller.stateKeyPath.unlink()
    with pytest.raises(ValidationError, match="key is missing"):
        controller.ensureStateKey()
    assert not controller.stateKeyPath.exists()


def testNewTrainingSettingsRemainCompatibleWithOlderProjects():
    old = Project.create("Older settings").toData()
    for name in ("blocksToSwap", "optimizer", "vaeChunkSize", "keepCheckpoints"):
        old["training"].pop(name)
    restored = Project.fromData(old)
    assert restored.training.blocksToSwap == 0
    custom = dataclasses.replace(restored.training, blocksToSwap=18, optimizer="AdamW8bit", vaeChunkSize=64, keepCheckpoints=2)
    low = custom.usePreset("Low")
    assert (low.blocksToSwap, low.optimizer, low.vaeChunkSize, low.keepCheckpoints) == (18, "AdamW8bit", 64, 2)
    assert (low.epochs, low.repeats) == (5, 5)


@pytest.mark.parametrize("changes", [{"blocksToSwap": 27}, {"blocksToSwap": -1}, {"vaeChunkSize": 127}, {"optimizer": "arbitrary.Module"}, {"keepCheckpoints": 0}])
def testInvalidBackendSettingsAreRejected(changes):
    with pytest.raises(ValidationError):
        dataclasses.replace(TrainingSettings(), **changes).validate()


def testMissingRuntimeCannotStartOrImportArbitraryStates(trainingInputs):
    workspace, project, models, controller, store, record, folder = trainingInputs
    with pytest.raises(ValidationError, match="runtime"):
        controller.start(project)
    assert not controller.busy
    assert not (controller.runtime.root / ".venv").exists()
    assert readJson(folder / "run.json")["status"] == "preparing"
