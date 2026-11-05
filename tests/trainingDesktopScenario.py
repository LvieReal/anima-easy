import os
import pathlib
import subprocess
import sys

from dearpygui import dearpygui

from animaEasy.errors import ValidationError
from animaEasy.trainingRuntime import RuntimeManager
from animaEasy.ui.shell import Desktop
from animaEasy.workspace import Workspace
from desktopScenario import activate, pump


def accent(statePath: pathlib.Path) -> None:
    workspace = Workspace(statePath)
    workspace.configure(str(statePath / "workspace"), setupComplete=True)
    desktop = Desktop(workspace)
    try:
        desktop.start()
        activate("navWorkspace")
        pump(desktop)
        window = subprocess.check_output(["xdotool", "search", "--all", "--pid", str(os.getpid()), "--name", "^Anima Easy$"], text=True).splitlines()[0]
        subprocess.run(["xdotool", "windowfocus", window], check=True)
        for color in ((40, 160, 224), (1, 0, 1), (218, 236, 248)):
            x, y = dearpygui.get_item_rect_min("accentColorInput")
            subprocess.run(["xdotool", "mousemove", "--window", window, str(round(x + 45)), str(round(y + 12)), "click", "1"], check=True)
            pump(desktop)
            subprocess.run(["xdotool", "keydown", "ctrl", "keydown", "a"], check=True)
            pump(desktop)
            subprocess.run(["xdotool", "keyup", "a", "keyup", "ctrl"], check=True)
            pump(desktop)
            value = "#" + "".join(f"{channel:02X}" for channel in color)
            keyboard = subprocess.Popen(["xdotool", "type", "--delay", "70", value])
            pump(desktop, lambda keyboard=keyboard: keyboard.poll() is not None)
            assert keyboard.returncode == 0
            subprocess.run(["xdotool", "key", "Return"], check=True)
            pump(desktop, lambda color=color: workspace.preferences.accentColor == color)
            assert tuple(round(channel) for channel in dearpygui.get_value("accentColorInput")[:3]) == color
            assert Workspace(statePath).preferences.accentColor == color
        assert "torch" not in sys.modules
    finally:
        desktop.close()


def trainingUi(statePath: pathlib.Path) -> None:
    workspace = Workspace(statePath)
    workspace.configure(str(statePath / "workspace"), setupComplete=True)
    project = workspace.projects.create("Training controls")
    desktop = Desktop(workspace)
    try:
        desktop.start()
        activate("navTraining")
        pump(desktop)
        assert desktop.view.section == "training"
        assert dearpygui.get_item_configuration("setupTrainingRuntime")["enabled"]
        assert not dearpygui.get_item_configuration("checkTrainingGpu")["enabled"]
        assert not dearpygui.get_item_configuration("stopTrainingJob")["enabled"]
        assert dearpygui.get_item_configuration("trainingLogPanel")["horizontal_scrollbar"]
        assert "No active" in dearpygui.get_item_configuration("trainingProgress")["overlay"]
        activate("setupTrainingRuntime")
        pump(desktop)
        assert dearpygui.is_item_shown("confirmDialog")
        activate("cancelConfirmation")
        assert not desktop.training.runtime.root.exists()
        assert not desktop.training.busy
        desktop.openProject(project.id)
        desktop.view.selectStep(userData=2)
        dearpygui.set_value("advancedTraining", True)
        pump(desktop)
        for tag, value in (("trainingBlocksToSwap", 18), ("trainingOptimizer", "AdamW8bit"), ("trainingVaeChunkSize", 64), ("trainingKeepCheckpoints", 2)):
            dearpygui.set_value(tag, value)
            activate(tag)
        dearpygui.set_value("trainingPreset", "Low")
        activate("trainingPreset")
        assert desktop.view.project.training.blocksToSwap == 18
        assert desktop.view.project.training.optimizer == "AdamW8bit"
        assert desktop.view.project.training.vaeChunkSize == 64
        assert desktop.view.project.training.keepCheckpoints == 2
        desktop.view.selectStep(userData=3)
        pump(desktop)
        assert not dearpygui.get_item_configuration("startTrainingButton")["enabled"]
        desktop.showTraining(project.id)
        pump(desktop)
        assert dearpygui.get_item_rect_size("trainingLogPanel")[0] > 450
        dearpygui.set_value("trainingLog", "\n".join(f"Renderer log line {index}" for index in range(100)))
        pump(desktop, lambda: dearpygui.get_y_scroll("trainingLogPanel") > 0 and dearpygui.get_y_scroll("trainingLogPanel") >= dearpygui.get_y_scroll_max("trainingLogPanel") - 2)
        assert "torch" not in sys.modules and "accelerate" not in sys.modules
        saved = workspace.projects.load(project.id)
        assert (saved.training.epochs, saved.training.repeats) == (5, 5)
        assert saved.training.blocksToSwap == 18
    finally:
        desktop.close()


def managedOperation(statePath: pathlib.Path) -> None:
    workspace = Workspace(statePath)
    workspace.configure(str(statePath / "workspace"), setupComplete=True)
    desktop = Desktop(workspace)
    desktop.training.runtime = RuntimeManager(statePath, pathlib.Path(os.environ["ANIMA_EASY_TEST_RUNTIME"]).resolve().parent)
    try:
        desktop.start()
        activate("navTraining")
        dearpygui.set_value("runtimeSetupPanel", True)
        pump(desktop)
        window = subprocess.check_output(["xdotool", "search", "--all", "--pid", str(os.getpid()), "--name", "^Anima Easy$"], text=True).splitlines()[0]
        subprocess.run(["xdotool", "windowfocus", window], check=True)

        def click(tag):
            x, y = dearpygui.get_item_rect_min(tag)
            width, height = dearpygui.get_item_rect_size(tag)
            subprocess.run(["xdotool", "mousemove", "--window", window, str(round(x + width / 2)), str(round(y + height / 2))], check=True)
            pump(desktop)
            subprocess.run(["xdotool", "mousedown", "1"], check=True)
            pump(desktop)
            subprocess.run(["xdotool", "mouseup", "1"], check=True)
            pump(desktop)

        click("checkTrainingGpu")
        pump(desktop, lambda: desktop.training.runner.process is not None)
        assert desktop.training.busy
        assert desktop.training.snapshot()["progress"] is None
        activate("navWorkspace")
        pump(desktop)
        previous = workspace.preferences.workspacePath
        dearpygui.set_value("settingsWorkspacePath", str(statePath / "other workspace"))
        try:
            desktop.view.saveWorkspace()
        except ValidationError as error:
            assert "Stop" in str(error)
        else:
            raise AssertionError("Changing workspaces during a managed operation must be blocked.")
        assert workspace.preferences.workspacePath == previous
        activate("navTraining")
        pump(desktop, lambda: dearpygui.get_item_configuration("stopTrainingJob")["enabled"])
        click("stopTrainingJob")
        assert dearpygui.does_item_exist("confirmDialog")
        click("acceptConfirmation")
        pump(desktop, lambda: not desktop.training.busy)
        assert desktop.training.snapshot()["status"] == "canceled", desktop.training.snapshot()
        assert desktop.training.runner.process is None
        assert desktop.training.snapshot()["progress"] is None
        assert "torch" not in sys.modules and "accelerate" not in sys.modules
    finally:
        desktop.close()


if __name__ == "__main__":
    {"accent": accent, "training": trainingUi, "managed": managedOperation}[sys.argv[1]](pathlib.Path(sys.argv[2]))
