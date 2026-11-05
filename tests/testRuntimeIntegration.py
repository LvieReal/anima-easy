import dataclasses
import os
import pathlib
import secrets
import shutil
import subprocess
import sys
import threading
import time

import PIL.Image
import pytest

from animaEasy.domain import Project
from animaEasy.errors import ValidationError
from animaEasy.storage import readJson
from animaEasy.trainingArguments import trainingArguments
from animaEasy.trainingAssets import runtimeContract
from animaEasy.trainingProcess import ProcessRunner
from animaEasy.trainingRuns import verifyInputs
from animaEasy.trainingRuntime import ASSET_FOLDER, RuntimeManager
from animaEasy.ui.trainingView import logTail
from testTrainingInputs import captureFixture


pytestmark = [pytest.mark.runtime, pytest.mark.skipif(not os.environ.get("ANIMA_EASY_TEST_RUNTIME"), reason="Set ANIMA_EASY_TEST_RUNTIME to an installed managed runtime to run real backend CPU contracts.")]


def backendRoot() -> pathlib.Path:
    root = pathlib.Path(os.environ["ANIMA_EASY_TEST_RUNTIME"]).resolve()
    assert (root / "trainer" / "anima_train_network.py").is_file()
    return root


def runContract(folder: pathlib.Path, specification: dict) -> dict:
    specificationPath = folder / "cpuSpecification.json"
    runtimeContract.writeData(specificationPath, specification)
    command = [shutil.which("uv"), "run", "--locked", "--no-sync", "--project", str(backendRoot()), "python", "-I", str(pathlib.Path(__file__).with_name("runtimeContractScenario.py")), str(specificationPath), str(ASSET_FOLDER)]
    runtime = RuntimeManager(backendRoot().parent.parent, backendRoot().parent)
    runtime.prepareProcessFolders()
    environment = runtime.environment()
    result = subprocess.run(command, cwd=backendRoot() / "trainer", env=environment, capture_output=True, text=True, encoding="utf-8", timeout=120)
    (folder / "cpuContract.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    return runtimeContract.readData(folder / "cpuContractResult.json")


def testActualAnimaBackendDatasetLoraCheckpointAndOptimizerResume(trainingInputs):
    workspace, project, models, controller, store, record, folder = trainingInputs
    dataset = pathlib.Path(project.datasetPath)
    PIL.Image.new("RGB", (64, 96), (35, 80, 180)).save(dataset / "image0.jpg")
    poison = dataset / "image0_0064x0096_anima.npz"
    poison.write_bytes(b"not a valid cache; it must never be read")
    recipe = captureFixture(trainingInputs)
    originals = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in dataset.iterdir()}
    specification = {"runId": record["id"], "runFolder": str(folder), "recipeHash": record["recipeHash"], "stateKeyPath": str(controller.stateKeyPath), "trainerPath": str(backendRoot() / "trainer"), "arguments": trainingArguments(folder, recipe), "token": secrets.token_hex(24), "resume": None}
    first = runContract(folder, specification)
    assert runtimeContract.EVENT_PREFIX not in logTail(folder / "cpuContract.log")
    assert "Saving a LoRA checkpoint" in logTail(folder / "cpuContract.log")
    assert first["completedEpochs"] == 1
    assert first["loraModules"] > 0
    assert first["checkpointName"] == "anima-000001.safetensors"
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == original for path, original in originals.items())
    states = store.states(record)
    assert len(states) == 1
    assert store.checkpoints(record) == [first["checkpointName"]]
    statePath = folder / "outputs" / first["stateName"]
    with pytest.raises(ValueError, match="authenticated"):
        runtimeContract.verifyState(statePath, record["id"], record["recipeHash"], secrets.token_bytes(32))
    updated = dataclasses.replace(project, training=dataclasses.replace(project.training, rank=64, epochs=30))
    workspace.projects.save(updated)
    (dataset / "image0.txt").write_text("later caption edit must not replace captured text", encoding="utf-8")
    captured = verifyInputs(folder, record["recipeHash"], controller.runtime.fingerprint, threading.Event(), lambda message: None)
    originalProject = Project.fromData(captured["project"])
    assert originalProject.training.rank == 20
    child = store.create(originalProject, controller.runtime.fingerprint, {"runId": record["id"], "stateName": first["stateName"]})
    childFolder = store.folder(project.id, child["id"])
    store.copySnapshot(folder, childFolder)
    resume = {"path": str(statePath), "runId": record["id"], "completedEpochs": states[0]["completedEpochs"], "completedSteps": states[0]["completedSteps"]}
    childSpecification = {**specification, "runId": child["id"], "runFolder": str(childFolder), "arguments": trainingArguments(childFolder, captured, resume), "resume": resume, "expectedPath": str(folder / "cpuExpected.json")}
    second = runContract(childFolder, childSpecification)
    assert second["completedEpochs"] == 2
    assert second["checkpointName"] == "anima-000002.safetensors"
    assert second["stateName"] == "anima-000002-state"
    assert runtimeContract.fileDigest(childFolder / "inputs" / "recipe.json") == record["recipeHash"]
    assert second["finalStateName"] == "anima-state"
    child["recipeHash"] = record["recipeHash"]
    store.save(child)
    assert store.checkpoints(child) == ["anima.safetensors"]
    assert [item["completedEpochs"] for item in store.states(child)] == [3, 2]
    with pytest.raises(ValidationError, match="already reached"):
        trainingArguments(childFolder, captured, {"completedEpochs": 3})
    optimizerPath = statePath / "optimizer.bin"
    originalOptimizer = optimizerPath.read_bytes()
    optimizerPath.write_bytes(originalOptimizer[:-1] + bytes([originalOptimizer[-1] ^ 1]))
    with pytest.raises(ValueError, match="changed or is incomplete"):
        runtimeContract.verifyState(statePath, record["id"], record["recipeHash"], controller.stateKeyPath.read_bytes())
    optimizerPath.write_bytes(originalOptimizer)
    manifestPath = statePath / runtimeContract.STATE_MARKER
    manifest = runtimeContract.readData(manifestPath)
    manifest["completedSteps"] += 1
    runtimeContract.writeData(manifestPath, manifest)
    with pytest.raises(ValueError, match="authenticated"):
        runtimeContract.verifyState(statePath, record["id"], record["recipeHash"], controller.stateKeyPath.read_bytes())
    assert store.states(record) == []


def testRealNoCudaPreflightFailsWithoutStartingTraining(trainingInputs):
    workspace, project, models, controller, store, record, folder = trainingInputs
    controller.runtime = RuntimeManager(backendRoot().parent.parent, backendRoot().parent)
    assert controller.runtime.installed(), "Run managed setup again after changing runtime assets."
    hardware = controller.runtime.probe(controller.runner, controller.cancelEvent, controller.appendLine)
    if hardware.get("cudaReady"):
        pytest.skip("This negative preflight check requires a machine without a usable CUDA GPU.")
    controller.start(project)
    started = time.monotonic()
    while controller.busy and time.monotonic() - started < 120:
        controller.thread.join(timeout=1)
    assert not controller.busy
    result = controller.snapshot()
    assert result["status"] == "failed"
    assert "CUDA GPU" in result["message"]
    assert result["progress"] is None
    actual = store.load(project.id, result["run"]["id"])
    runFolder = store.folder(project.id, actual["id"])
    assert actual["status"] == "failed"
    assert readJson(runFolder / "inputs" / "hardware.json")["cudaReady"] is False
    assert not list((runFolder / "outputs").iterdir())
    assert "CUDA GPU" in (runFolder / "training.log").read_text(encoding="utf-8")


def testInstalledSourceAndBridgeIntegrityAreCheckedBeforeExecution():
    runtime = RuntimeManager(backendRoot().parent.parent, backendRoot().parent)
    cancel = threading.Event()
    runtime.verifyInstallation(cancel)
    for path in (runtime.trainerPath / "library" / "strategy_base.py", runtime.root / "runtimeBridge.py"):
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\nchanged after installation\n")
            with pytest.raises(ValidationError, match="changed"):
                runtime.verifyInstallation(cancel)
        finally:
            path.write_bytes(original)
    runtime.verifyInstallation(cancel)


@pytest.mark.desktop
@pytest.mark.skipif(sys.platform != "linux" or not os.environ.get("DISPLAY") or not shutil.which("xdotool"), reason="This managed-operation input check requires X11 and xdotool.")
def testNativeGpuCheckAndStopWithActualManagedProcess(sandboxFolder):
    from testDesktop import runScenario

    runScenario("managed", sandboxFolder / "state", script="trainingDesktopScenario.py")


def testFailedPythonPrerequisitePreservesAWorkingInstallation():
    runtime = RuntimeManager(backendRoot().parent.parent, backendRoot().parent)
    assert runtime.installed()
    paths = (runtime.readyPath, runtime.probePath, runtime.trainerPath / "anima_train_network.py", runtime.root / "runtimeBridge.py")
    originals = {path: path.read_bytes() if path.exists() else None for path in paths}
    configuration = runtime.root / "uv.toml"
    assert not configuration.exists()
    try:
        configuration.write_text("python-downloads = [invalid\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="Managed Python installation failed"):
            runtime.install(ProcessRunner(), threading.Event(), lambda line: None)
        assert runtime.installed()
        assert {path: path.read_bytes() if path.exists() else None for path in paths} == originals
    finally:
        configuration.unlink(missing_ok=True)
    runtime.verifyInstallation(threading.Event())
