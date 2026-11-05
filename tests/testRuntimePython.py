import json
import os
import pathlib
import shutil
import subprocess
import threading

import pytest

from animaEasy.errors import ValidationError
from animaEasy.trainingProcess import ProcessRunner
from animaEasy.trainingRuntime import RUNTIME_PYTHON, RuntimeManager, findUv


@pytest.fixture
def emptyRuntime(sandboxFolder):
    runtime = RuntimeManager(sandboxFolder / "state", sandboxFolder / "runtime storage")
    runtime.claimStorage()
    runtime.prepareProcessFolders()
    return runtime


@pytest.fixture
def seededRuntime(emptyRuntime):
    source = pathlib.Path(os.environ["ANIMA_EASY_TEST_RUNTIME"]) / "python"
    installations = [path for path in source.iterdir() if path.name.startswith(f"cpython-{RUNTIME_PYTHON}-") and path.is_dir()]
    assert len(installations) == 1
    destination = emptyRuntime.root / "python" / installations[0].name
    shutil.copytree(installations[0], destination, symlinks=True)
    return emptyRuntime, destination


def testStorageHardlinkProbeUsesTheFilesystemAndCleansUp(emptyRuntime):
    assert emptyRuntime.supportsHardlinks()
    assert list((emptyRuntime.root / "temp").iterdir()) == []
    assert not emptyRuntime.installed()


def testPythonBootstrapDoesNotRecoverFromAnActualUvConfigurationError(emptyRuntime):
    (emptyRuntime.root / "uv.toml").write_text("python-downloads = [invalid\n", encoding="utf-8")
    lines = []
    with pytest.raises(ValidationError, match="Managed Python installation failed"):
        emptyRuntime.preparePython(findUv(), ProcessRunner(), threading.Event(), lines.append)
    assert any("parse" in line.lower() for line in lines)
    assert not list((emptyRuntime.root / "python").iterdir())
    assert not (emptyRuntime.root / ".venv").exists()
    assert not emptyRuntime.readyPath.exists()


@pytest.mark.parametrize("operation", ["python", "dependencies"])
def testCanceledInstallationStagesDoNotLaunchAProcess(emptyRuntime, operation):
    cancel = threading.Event()
    cancel.set()
    runner = ProcessRunner()
    with pytest.raises(InterruptedError):
        if operation == "python":
            emptyRuntime.preparePython(findUv(), runner, cancel, lambda line: None)
        else:
            emptyRuntime.syncDependencies(findUv(), emptyRuntime.root / "python" / "not-launched", True, runner, cancel, lambda line: None)
    assert runner.process is None
    assert not emptyRuntime.logPath.exists()
    assert not (emptyRuntime.root / ".venv").exists()
    assert not emptyRuntime.readyPath.exists()


@pytest.mark.runtime
@pytest.mark.skipif(not os.environ.get("ANIMA_EASY_TEST_RUNTIME"), reason="Set ANIMA_EASY_TEST_RUNTIME to a real managed runtime to exercise its Python and a small locked package install.")
def testVersionedPythonAndRealPackageCopiesWithoutTheMinorAlias(seededRuntime):
    runtime, installation = seededRuntime
    runner = ProcessRunner()
    cancel = threading.Event()
    executable = findUv()
    python = runtime.preparePython(executable, runner, cancel, lambda line: None)
    assert python.is_relative_to(installation)
    minorAlias = installation.with_name(installation.name.replace(f"-{RUNTIME_PYTHON}-", "-3.12-", 1))
    if minorAlias.is_symlink():
        minorAlias.unlink()
    elif os.name == "nt" and minorAlias.is_junction():
        minorAlias.rmdir()
    assert not minorAlias.exists()
    (runtime.root / "pyproject.toml").write_text(f'[project]\nname = "runtime-copy-check"\nversion = "0.0.0"\nrequires-python = "=={RUNTIME_PYTHON}"\ndependencies = ["filelock==4.2.9"]\n\n[tool.uv]\npackage = false\n', encoding="utf-8")
    command = [executable, "lock", "--project", str(runtime.root), "--python", str(python), "--no-python-downloads", "--default-index", "https://pypi.org/simple"]
    code = runner.run(command, runtime.root, runtime.environment(), runtime.logPath, cancel)
    assert code == 0, runtime.logPath.read_text(encoding="utf-8")
    runtime.syncDependencies(executable, python, True, runner, cancel, lambda line: None)
    script = runtime.root / "checkCopy.py"
    script.write_text("import filelock, importlib.metadata, json, pathlib, sys, tempfile\nprint(json.dumps({'version': importlib.metadata.version('filelock'), 'module': filelock.__file__, 'links': pathlib.Path(filelock.__file__).stat().st_nlink, 'base': str(pathlib.Path(sys.base_prefix).resolve()), 'prefix': sys.prefix, 'temp': tempfile.gettempdir()}))\n", encoding="utf-8")
    result = subprocess.run(runtime.command(script), cwd=runtime.root, env=runtime.environment(), capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["version"] == "4.2.9"
    assert pathlib.Path(report["module"]).is_relative_to(runtime.root / ".venv")
    assert report["links"] == 1
    assert pathlib.Path(report["base"]) == installation.resolve()
    assert pathlib.Path(report["prefix"]) == runtime.root / ".venv"
    assert pathlib.Path(report["temp"]) == runtime.root / "temp"
    assert not minorAlias.exists()
    cached = list((runtime.root / "uvCache" / "archive-v0").glob("*/filelock/__init__.py"))
    assert len(cached) == 1
    assert cached[0].read_bytes() == pathlib.Path(report["module"]).read_bytes()
    assert not os.path.samefile(cached[0], report["module"])
    assert not runtime.readyPath.exists()


@pytest.mark.runtime
@pytest.mark.skipif(not os.environ.get("ANIMA_EASY_TEST_RUNTIME"), reason="Set ANIMA_EASY_TEST_RUNTIME to a real managed runtime to test external-interpreter rejection.")
def testBootstrapRejectsARealInterpreterOutsidePrivateStorage(seededRuntime, sandboxFolder):
    runtime, installation = seededRuntime
    outside = sandboxFolder / installation.name
    shutil.copytree(installation, outside, symlinks=True)
    executable = installation / ("python.exe" if os.name == "nt" else "bin/python3.12")
    executable.unlink()
    try:
        executable.symlink_to(outside / executable.relative_to(installation))
    except OSError:
        pytest.skip("This account cannot create symbolic links.")
    with pytest.raises(ValidationError, match="inside the selected runtime storage"):
        runtime.preparePython(findUv(), ProcessRunner(), threading.Event(), lambda line: None)
    assert not runtime.readyPath.exists()
    assert not (runtime.root / ".venv").exists()


@pytest.mark.runtime
@pytest.mark.skipif(not os.environ.get("ANIMA_EASY_TEST_LINKLESS_RUNTIME"), reason="Set ANIMA_EASY_TEST_LINKLESS_RUNTIME to a disposable linkless-volume runtime with the exact managed Python already downloaded.")
@pytest.mark.parametrize("cancelOnLinkFailure", [False, True])
def testRealLinklessVolumeBootstrapAndCancellation(cancelOnLinkFailure):
    root = pathlib.Path(os.environ["ANIMA_EASY_TEST_LINKLESS_RUNTIME"]).resolve()
    runtime = RuntimeManager(root.parent.parent / "state", root.parent)
    runtime.claimStorage()
    runtime.prepareProcessFolders()
    assert not runtime.supportsHardlinks()
    assert not runtime.installed()
    cancel = threading.Event()
    lines = []
    def received(line):
        lines.append(line)
        if cancelOnLinkFailure and "Failed to create Python minor version link directory" in line:
            cancel.set()
    if cancelOnLinkFailure:
        with pytest.raises(InterruptedError):
            runtime.preparePython(findUv(), ProcessRunner(), cancel, received)
        assert cancel.is_set()
    else:
        python = runtime.preparePython(findUv(), ProcessRunner(), cancel, received)
        assert python.is_relative_to(root / "python")
        assert any("continuing directly" in line for line in lines)
        assert "continuing directly" in runtime.logPath.read_text(encoding="utf-8")
        script = "import json, platform, sys, tempfile; print(json.dumps({'version': platform.python_version(), 'base': sys.base_prefix, 'python': sys.executable, 'temp': tempfile.gettempdir()}))"
        command = [findUv(), "run", "--no-project", "--no-python-downloads", "--python", str(python), "python", "-I", "-c", script]
        result = subprocess.run(command, cwd=root, env=runtime.environment(), capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout)
        assert report["version"] == RUNTIME_PYTHON
        assert pathlib.Path(report["base"]).is_relative_to(root / "python")
        assert pathlib.Path(report["python"]).is_relative_to(root / "python")
        assert pathlib.Path(report["temp"]) == root / "temp"
    assert not runtime.readyPath.exists()
