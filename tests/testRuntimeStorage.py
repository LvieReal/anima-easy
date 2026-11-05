import dataclasses
import json
import pathlib
import subprocess
import sys
import threading

import pytest

from animaEasy.errors import StorageError, ValidationError
from animaEasy.storage import readJson, writeJson
from animaEasy.training import TrainingController
from animaEasy.trainingRuntime import ASSET_FOLDER, RUNTIME_ID, RuntimeManager, findUv
from animaEasy.workspace import Workspace
from testTrainingProcess import command
from animaEasy.trainingProcess import trainingEnvironment


def testRuntimeStoragePersistsWithoutMovingExistingData(trainingInputs):
    workspace, project, models, controller, store, record, folder = trainingInputs
    project = workspace.projects.save(dataclasses.replace(project, training=dataclasses.replace(project.training, resolution=896)))
    previousRoot = controller.runtime.root
    previousRoot.mkdir(parents=True)
    sentinel = previousRoot / "keep.txt"
    sentinel.write_text("old installation stays here", encoding="utf-8")
    key = controller.stateKeyPath.read_bytes()
    identity = controller.stateKeyPath.with_name("stateIdentity.json").read_bytes()
    fingerprint = controller.runtime.fingerprint
    storage = workspace.statePath.parent / "other drive" / "training storage"
    controller.setRuntimeStorage(str(storage))
    assert controller.runtime.root == storage / RUNTIME_ID
    assert controller.runtime.fingerprint == fingerprint
    assert not controller.runtime.root.exists()
    assert sentinel.read_text(encoding="utf-8") == "old installation stays here"
    assert controller.stateKeyPath.read_bytes() == key
    assert controller.stateKeyPath.with_name("stateIdentity.json").read_bytes() == identity
    workspace.setAppearance("Light", (220, 100, 80))
    reopened = Workspace(workspace.statePath)
    second = TrainingController(reopened)
    try:
        assert second.runtime.root == storage / RUNTIME_ID
        assert reopened.projects.load(project.id).training.resolution == 896
        assert reopened.models.records == workspace.models.records
        assert store.load(project.id, record["id"])["id"] == record["id"]
        assert folder.is_dir()
    finally:
        second.close()


def testUnrecognizedRuntimeFolderIsNotOverwritten(sandboxFolder):
    workspace = Workspace(sandboxFolder / "state")
    workspace.configure(str(sandboxFolder / "workspace"), setupComplete=True)
    previous = workspace.preferencesPath.read_bytes()
    storage = sandboxFolder / "other storage"
    runtime = RuntimeManager(workspace.statePath, storage)
    path = runtime.trainerPath / "valuable.txt"
    path.parent.mkdir(parents=True)
    path.write_text("not an application runtime", encoding="utf-8")
    controller = TrainingController(workspace)
    try:
        with pytest.raises(ValidationError, match="already contains"):
            controller.setRuntimeStorage(str(storage))
        with pytest.raises(ValidationError, match="already contains"):
            runtime.claimStorage()
        assert workspace.preferencesPath.read_bytes() == previous
        assert path.read_text(encoding="utf-8") == "not an application runtime"
        assert not runtime.ownershipPath.exists()
    finally:
        controller.close()


@pytest.mark.parametrize("value", [None, 12, [], "relative/path", "\x00bad"])
def testInvalidRuntimePreferencesAreReportedWithoutReset(sandboxFolder, value):
    workspace = Workspace(sandboxFolder / "state")
    workspace.configure(str(sandboxFolder / "workspace"), setupComplete=True)
    data = readJson(workspace.preferencesPath)
    writeJson(workspace.preferencesPath, {**data, "runtimeStoragePath": value})
    before = workspace.preferencesPath.read_bytes()
    with pytest.raises(StorageError, match="runtime storage"):
        Workspace(workspace.statePath)
    assert workspace.preferencesPath.read_bytes() == before


def testOlderPreferencesKeepTheDefaultRuntimeLocation(sandboxFolder):
    state = sandboxFolder / "state"
    writeJson(state / "settings.json", {"schemaVersion": 1, "workspacePath": str(sandboxFolder / "workspace"), "setupComplete": True})
    controller = TrainingController(Workspace(state))
    try:
        assert controller.runtime.root == state / "training" / RUNTIME_ID
        assert not controller.runtime.root.exists()
    finally:
        controller.close()


def testOwnedPartialSetupCanBeReopenedButIsNotMarkedInstalled(sandboxFolder):
    runtime = RuntimeManager(sandboxFolder / "state", sandboxFolder / "storage")
    runtime.claimStorage()
    (runtime.root / "trainer.download").write_bytes(b"partial download")
    reopened = RuntimeManager(sandboxFolder / "state", sandboxFolder / "storage")
    reopened.claimStorage()
    reopened.validateStorage()
    assert not reopened.installed()
    assert not reopened.readyPath.exists()
    assert (runtime.root / "trainer.download").read_bytes() == b"partial download"


@pytest.mark.parametrize("damagedReadyFile", [False, True])
def testLegacyManagedProjectCanAcquireAnOwnershipMarker(sandboxFolder, damagedReadyFile):
    runtime = RuntimeManager(sandboxFolder / "state")
    runtime.root.mkdir(parents=True)
    (runtime.root / "pyproject.toml").write_bytes((ASSET_FOLDER / "runtimeProject.toml").read_bytes())
    if damagedReadyFile:
        runtime.readyPath.write_bytes(b"{damaged metadata")
    runtime.claimStorage()
    assert readJson(runtime.ownershipPath)["runtimeId"] == RUNTIME_ID
    assert not runtime.installed()


def testActualUvAndPythonTemporaryStorageUseTheSelectedFolder(sandboxFolder):
    runtime = RuntimeManager(sandboxFolder / "state", sandboxFolder / "drive with spaces" / "runtime")
    runtime.claimStorage()
    runtime.prepareProcessFolders()
    environment = runtime.environment()
    for arguments, expected in ((["cache", "dir"], runtime.root / "uvCache"), (["python", "dir"], runtime.root / "python")):
        result = subprocess.run([findUv(), *arguments], cwd=sandboxFolder, env=environment, capture_output=True, text=True, encoding="utf-8", check=True, timeout=20)
        assert pathlib.Path(result.stdout.strip()) == expected
    script = "import json, tempfile; directory = tempfile.TemporaryDirectory(); print(json.dumps({'base': tempfile.gettempdir(), 'created': directory.name})); directory.cleanup()"
    result = subprocess.run([findUv(), "run", "--no-project", "--python", sys.executable, "python", "-I", "-c", script], cwd=sandboxFolder, env=environment, capture_output=True, text=True, encoding="utf-8", check=True, timeout=20)
    paths = json.loads(result.stdout)
    assert pathlib.Path(paths["base"]) == runtime.root / "temp"
    assert pathlib.Path(paths["created"]).parent == runtime.root / "temp"
    assert not pathlib.Path(paths["created"]).exists()
    assert environment["UV_PYTHON_INSTALL_BIN"] == "0"
    assert environment["UV_PYTHON_INSTALL_REGISTRY"] == "0"
    assert environment["CUDA_CACHE_PATH"] == str(runtime.root / "cudaCache")


def testActiveOperationBlocksRelocationAndAnotherProfileUsingTheSameRuntime(sandboxFolder):
    storage = sandboxFolder / "shared runtime storage"
    first = TrainingController(Workspace(sandboxFolder / "state one"))
    second = TrainingController(Workspace(sandboxFolder / "state two"))
    first.setRuntimeStorage(str(storage))
    second.setRuntimeStorage(str(storage))
    ready = threading.Event()
    entered = threading.Event()
    stop = sandboxFolder / "stop.request"
    def operation():
        first.runner.run(command("cooperative", str(stop)), sandboxFolder, trainingEnvironment(), sandboxFolder / "process.log", first.cancelEvent, lambda line: ready.set() if line == "READY" else None, stopPath=stop)
    try:
        first.launch("check", operation)
        assert ready.wait(10)
        with pytest.raises(ValidationError, match="Stop"):
            first.setRuntimeStorage(str(sandboxFolder / "different storage"))
        second.launch("check", entered.set)
        second.thread.join(timeout=10)
        assert not second.busy and second.snapshot()["status"] == "failed"
        assert not entered.is_set()
        assert "Another application instance" in second.snapshot()["message"]
        assert first.busy
    finally:
        first.close()
        second.close()
    assert "COOPERATIVE_STOP" in (sandboxFolder / "process.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["setup.log", "trainer.download"])
def testRuntimeWritableFilesCannotRedirectToExistingUserFiles(sandboxFolder, name):
    runtime = RuntimeManager(sandboxFolder / "state", sandboxFolder / "storage")
    runtime.root.mkdir(parents=True)
    original = sandboxFolder / "valuable.bin"
    original.write_bytes(b"keep these bytes")
    try:
        (runtime.root / name).symlink_to(original)
    except (OSError, NotImplementedError):
        pytest.skip("Creating a symbolic link is not permitted in this environment.")
    with pytest.raises(ValidationError, match="symbolic links"):
        runtime.claimStorage()
    assert original.read_bytes() == b"keep these bytes"
    assert not runtime.ownershipPath.exists()
