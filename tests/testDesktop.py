import os
import pathlib
import shutil
import subprocess
import sys

import PIL.Image
import pytest

from animaEasy.workspace import Workspace


pytestmark = [pytest.mark.desktop, pytest.mark.skipif(sys.platform == "linux" and not os.environ.get("DISPLAY"), reason="A real X11/XWayland display or Xvfb is required.")]


def runScenario(name: str, *paths: pathlib.Path, script: str = "desktopScenario.py") -> None:
    script = pathlib.Path(__file__).with_name(script)
    command = [shutil.which("uv"), "run", "--no-sync", "python", "-X", "faulthandler", str(script), name, *map(str, paths)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr


def testRealDesktopWorkflowAndRestart(sandboxFolder, sampleDataset, sampleModel):
    statePath = sandboxFolder / "state"
    # NOTE: Dear PyGui owns process-global graphics state; restart it in a fresh process.
    runScenario("workflow", statePath, sampleDataset, sampleModel)
    runScenario("restart", statePath)
    workspace = Workspace(statePath)
    assert workspace.projects.list().projects[0].notes == "Saved when closing immediately"


def testModelImportFailureStaysInTheUi(sandboxFolder):
    runScenario("error", sandboxFolder / "state")


def testShutdownConflictPreservesRecoveryDraft(sandboxFolder):
    runScenario("recovery", sandboxFolder / "state")


def testCaptionModesBulkActionsAndFreshDesktopReopen(sandboxFolder, sampleDataset):
    statePath = sandboxFolder / "state"
    runScenario("captions", statePath, sampleDataset, script="updateDesktopScenario.py")
    runScenario("reopen", statePath, script="updateDesktopScenario.py")


def testAppearancePresetsAndActualHintHover(sandboxFolder):
    runScenario("appearance", sandboxFolder / "state", script="updateDesktopScenario.py")


def testLargeImageListPaginationKeepsBulkSelections(sandboxFolder):
    datasetPath = sandboxFolder / "many images"
    datasetPath.mkdir()
    for index in range(103):
        PIL.Image.new("RGB", (32, 48), (85, 105, 170)).save(datasetPath / f"image{index:03d}.png")
    runScenario("pagination", sandboxFolder / "state", datasetPath, script="updateDesktopScenario.py")


@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("xdotool"), reason="Real picker input uses X11 keyboard and mouse events.")
def testCustomAccentThroughRealHexKeyboardInput(sandboxFolder):
    runScenario("accent", sandboxFolder / "state", script="trainingDesktopScenario.py")


def testNativeTrainingControlsAndExplicitSetupConfirmation(sandboxFolder):
    runScenario("training", sandboxFolder / "state", script="trainingDesktopScenario.py")


@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("xdotool"), reason="Native file selection and dialog color checks use X11 input.")
def testNativeFileSelectionPreservesExtensionsAndDialogsUseTheAccent(sandboxFolder, sampleModel):
    folder = sandboxFolder / "model files"
    folder.mkdir()
    path = folder / "model.v1.café.safetensors"
    shutil.copyfile(sampleModel, path)
    runScenario("files", sandboxFolder / "state", path, script="dialogDesktopScenario.py")


@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("xdotool"), reason="Native runtime-folder selection uses X11 input.")
def testNativeRuntimeLocationIsSavedBeforeSetupConfirmation(sandboxFolder):
    runScenario("location", sandboxFolder / "state", script="dialogDesktopScenario.py")
