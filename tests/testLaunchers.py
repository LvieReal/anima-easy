import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from animaEasy import VERSION


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("argument, expected", [("--help", "--data-dir"), ("--version", f"Anima Easy {VERSION}")])
def testPlatformLauncherFromAnotherWorkingDirectory(sandboxFolder, argument, expected):
    folder = sandboxFolder / "working directory with spaces"
    folder.mkdir()
    if sys.platform == "win32":
        script = PROJECT_ROOT / "runWindows.bat"
        command = f'cmd.exe /d /s /c ""{script}" {argument}"'
    else:
        command = [str(PROJECT_ROOT / "runLinux.sh"), argument]
    result = subprocess.run(command, cwd=folder, capture_output=True, text=True, encoding="utf-8", input="", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="Linux launcher behavior")
def testLinuxLauncherThroughAPathWithSpaces(sandboxFolder):
    link = sandboxFolder / "repository with spaces"
    link.symlink_to(PROJECT_ROOT, target_is_directory=True)
    result = subprocess.run([str(link / "runLinux.sh"), "--version"], cwd=sandboxFolder, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Anima Easy {VERSION}" in result.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="Linux launcher behavior")
def testLinuxLauncherExplainsMissingUvWithoutInstallingAnything(sandboxFolder):
    if shutil.which("uv", path="/usr/bin:/bin") is not None:
        pytest.skip("A system-wide uv installation is present.")
    environment = {**os.environ, "HOME": str(sandboxFolder), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(["/bin/bash", str(PROJECT_ROOT / "runLinux.sh"), "--version"], env=environment, cwd=sandboxFolder, capture_output=True, text=True, timeout=10)
    assert result.returncode == 127
    assert "uv was not found" in result.stderr
    assert not list(sandboxFolder.iterdir())
