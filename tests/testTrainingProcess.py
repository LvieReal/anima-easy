import os
import pathlib
import shutil
import sys
import threading
import time

import psutil
import pytest

from animaEasy.errors import ValidationError
from animaEasy.training import TrainingController
from animaEasy.trainingProcess import ProcessRunner, trainingEnvironment
from animaEasy.workspace import Workspace


def command(mode: str, *arguments: str) -> list[str]:
    return [shutil.which("uv"), "run", "--no-project", "--python", sys.executable, "python", "-u", str(pathlib.Path(__file__).with_name("processScenario.py")), mode, *arguments]


def testRealProcessStreamsAreCapturedWithoutShellInterpretation(sandboxFolder):
    runner = ProcessRunner()
    lines = []
    literal = 'spaces & ; | " café'
    logPath = sandboxFolder / "full log.txt"
    code = runner.run(command("streams", literal), sandboxFolder, trainingEnvironment(), logPath, threading.Event(), lines.append)
    assert code == 7
    assert f"ARGUMENT={literal}" in lines
    assert f"CWD={sandboxFolder}" in lines
    assert "stderr output" in lines
    assert "red" in lines
    assert "café" in lines
    assert all(len(line) <= 16384 for line in lines)
    assert b"x" * 70000 in logPath.read_bytes()
    assert "café" in logPath.read_text(encoding="utf-8")
    assert runner.process is None


def testCooperativeStopWaitsForChildAcknowledgment(sandboxFolder):
    runner = ProcessRunner()
    cancel = threading.Event()
    lines = []
    stopPath = sandboxFolder / "stop.request"
    def received(line):
        lines.append(line)
        if line == "READY":
            cancel.set()
    code = runner.run(command("cooperative", str(stopPath)), sandboxFolder, trainingEnvironment(), sandboxFolder / "log.txt", cancel, received, stopPath=stopPath, graceSeconds=3)
    assert code == 130
    assert "COOPERATIVE_STOP" in lines
    assert stopPath.exists()
    assert runner.process is None


def testCancellationTerminatesTheOwnedProcessTree(sandboxFolder):
    runner = ProcessRunner()
    cancel = threading.Event()
    processes = {}
    def received(line):
        if line.startswith(("ROOT=", "LEAF=")):
            role, processId = line.split("=")
            processes[role] = psutil.Process(int(processId))
            if len(processes) == 2:
                cancel.set()
    started = time.monotonic()
    runner.run(command("tree"), sandboxFolder, trainingEnvironment(), sandboxFolder / "log.txt", cancel, received, timeoutSeconds=10)
    assert time.monotonic() - started < 10
    assert set(processes) == {"ROOT", "LEAF"}
    for process in processes.values():
        assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE


def testSilentChildHasABoundedTimeout(sandboxFolder):
    runner = ProcessRunner()
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="time limit"):
        runner.run(command("silent"), sandboxFolder, trainingEnvironment(), sandboxFolder / "log.txt", threading.Event(), timeoutSeconds=0.3)
    assert time.monotonic() - started < 5
    assert runner.process is None


def testCancellationBeforeLaunchDoesNotCreateAChildOrLog(sandboxFolder):
    runner = ProcessRunner()
    canceled = threading.Event()
    canceled.set()
    with pytest.raises(InterruptedError):
        runner.run(command("silent"), sandboxFolder, trainingEnvironment(), sandboxFolder / "log.txt", canceled)
    assert runner.process is None
    assert not (sandboxFolder / "log.txt").exists()


def testOutputHandlerFailureStopsTheRealChild(sandboxFolder):
    runner = ProcessRunner()
    def failed(line):
        raise OSError("The output consumer failed")
    with pytest.raises(RuntimeError, match="capture training output"):
        runner.run(command("cooperative", str(sandboxFolder / "stop")), sandboxFolder, trainingEnvironment(), sandboxFolder / "log.txt", threading.Event(), failed)
    assert runner.process is None


def testControllerOwnsOneOperationAndClosingStopsItsRealChild(sandboxFolder):
    controller = TrainingController(Workspace(sandboxFolder / "state"))
    ready = threading.Event()
    def operation():
        controller.runner.run(command("cooperative", str(sandboxFolder / "stop")), sandboxFolder, trainingEnvironment(), sandboxFolder / "log.txt", controller.cancelEvent, lambda line: ready.set() if line == "READY" else None, stopPath=sandboxFolder / "stop")
    try:
        controller.launch("check", operation)
        assert ready.wait(10)
        with pytest.raises(ValidationError, match="One managed job"):
            controller.launch("check", operation)
        controller.close()
        assert not controller.busy
        assert not controller.thread.is_alive()
        assert controller.runner.process is None
        assert "COOPERATIVE_STOP" in (sandboxFolder / "log.txt").read_text(encoding="utf-8")
    finally:
        controller.close()


def testManagedEnvironmentDoesNotLeakPythonOrDistributedLaunchState():
    environment = trainingEnvironment()
    assert "PYTHONPATH" not in environment
    assert "VIRTUAL_ENV" not in environment
    assert "LOCAL_RANK" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["PYTHONUTF8"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == os.environ.get("CUDA_VISIBLE_DEVICES", "0")
