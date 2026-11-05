import os
import pathlib
import stat
import threading
import zipfile

import pytest

from animaEasy.errors import ValidationError
from animaEasy.trainingProcess import ProcessRunner
from animaEasy.trainingRuntime import TRAINER_REVISION, RuntimeManager, extractTrainer, runtimeFingerprint


def testRuntimeStartsUninstalledAndDoesNotInstallOnInspection(sandboxFolder):
    runtime = RuntimeManager(sandboxFolder / "state")
    assert not runtime.installed()
    assert runtime.probeResult() == {}
    assert not runtime.root.exists()
    assert len(runtimeFingerprint()) == 64


@pytest.mark.parametrize("name", ["/outside.py", "../outside.py", f"sd-scripts-{TRAINER_REVISION}/../../outside.py", f"sd-scripts-{TRAINER_REVISION}/..\\outside.py", "wrong-root/train.py"])
def testArchiveExtractionRejectsTraversal(sandboxFolder, name):
    archive = sandboxFolder / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(name, "unsafe")
    with pytest.raises(ValidationError, match="unsafe path"):
        extractTrainer(archive, sandboxFolder / "trainer")
    assert not (sandboxFolder.parent / "outside.py").exists()


def testArchiveExtractionRejectsSymbolicLinks(sandboxFolder):
    archive = sandboxFolder / "link.zip"
    entry = zipfile.ZipInfo(f"sd-scripts-{TRAINER_REVISION}/link")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(entry, "../../elsewhere")
    with pytest.raises(ValidationError, match="unsafe path"):
        extractTrainer(archive, sandboxFolder / "trainer")


def testArchiveExtractionPreservesContentsInTheManagedFolder(sandboxFolder):
    archive = sandboxFolder / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(f"sd-scripts-{TRAINER_REVISION}/library/example.py", "value = 3\n")
    destination = sandboxFolder / "trainer"
    extractTrainer(archive, destination)
    assert (destination / "library" / "example.py").read_text(encoding="utf-8") == "value = 3\n"
    assert sorted(path.relative_to(destination).as_posix() for path in destination.rglob("*")) == ["library", "library/example.py"]


def testManagedCommandUsesTheIsolatedProjectAndLiteralPaths(sandboxFolder):
    runtime = RuntimeManager(sandboxFolder / "state with spaces")
    script = pathlib.Path("/literal path/runtimeBridge.py")
    result = runtime.command(script, "a quoted argument & ;")
    assert result[result.index("--project") + 1] == str(runtime.root)
    assert result[-1] == "a quoted argument & ;"
    assert "--no-sync" in result and "-I" in result
    assert runtime.environment()["UV_PROJECT_ENVIRONMENT"] == str(runtime.root / ".venv")


def testSetupChecksTheActualUvExecutableWithoutInstallingDependencies(sandboxFolder):
    runtime = RuntimeManager(sandboxFolder / "state", sandboxFolder / "storage with spaces")
    runtime.claimStorage()
    runtime.prepareProcessFolders()
    lines = []
    executable = runtime.checkUv(ProcessRunner(), threading.Event(), lines.append)
    assert pathlib.Path(executable).is_file()
    assert any(line.startswith("uv ") for line in lines)
    assert "uv " in runtime.logPath.read_text(encoding="utf-8")
    assert not (runtime.root / ".venv").exists()
    assert not runtime.readyPath.exists()
    assert not runtime.trainerPath.exists()


@pytest.mark.skipif(not os.environ.get("ANIMA_EASY_TEST_OLD_UV"), reason="Set ANIMA_EASY_TEST_OLD_UV to a real uv 0.11.25 executable to check the outdated-installer regression.")
def testSetupRejectsTheActualOldUvBeforeReplacingRuntimeFiles(sandboxFolder, monkeypatch):
    executable = pathlib.Path(os.environ["ANIMA_EASY_TEST_OLD_UV"])
    assert executable.is_file()
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])
    runtime = RuntimeManager(sandboxFolder / "state", sandboxFolder / "storage with spaces")
    runtime.claimStorage()
    runtime.trainerPath.mkdir()
    original = runtime.trainerPath / "preserve.txt"
    original.write_bytes(b"existing runtime files must survive a failed prerequisite check")
    runtime.readyPath.write_bytes(b"existing readiness file")
    lines = []
    with pytest.raises(ValidationError, match=r"uv 0\.11\.25 is too old") as caught:
        runtime.install(ProcessRunner(), threading.Event(), lines.append)
    assert "uv self update" in str(caught.value)
    assert "3.12.16" in str(caught.value)
    assert "0.13.7" in str(caught.value)
    assert "uv 0.11.25" in runtime.logPath.read_text(encoding="utf-8")
    assert runtime.readyPath.read_bytes() == b"existing readiness file"
    assert original.read_bytes() == b"existing runtime files must survive a failed prerequisite check"
    assert not (runtime.root / ".venv").exists()
    assert not (runtime.root / "trainer.zip").exists()
    assert not (runtime.root / "trainer.pending").exists()


def testCanceledUvCheckDoesNotLaunchAProcess(sandboxFolder):
    runtime = RuntimeManager(sandboxFolder / "state")
    runtime.claimStorage()
    cancel = threading.Event()
    cancel.set()
    runner = ProcessRunner()
    with pytest.raises(InterruptedError):
        runtime.checkUv(runner, cancel, lambda line: None)
    assert runner.process is None
    assert not runtime.logPath.exists()
