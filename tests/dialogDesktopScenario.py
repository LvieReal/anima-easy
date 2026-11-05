import os
import pathlib
import subprocess
import sys

import PIL.Image
from dearpygui import dearpygui

from animaEasy.ui import theme
from animaEasy.ui.modelsView import ModelsView
from animaEasy.ui.shell import Desktop
from animaEasy.workspace import Workspace
from desktopScenario import activate, pump


def windowId() -> str:
    window = subprocess.check_output(["xdotool", "search", "--all", "--pid", str(os.getpid()), "--name", "^Anima Easy$"], text=True).splitlines()[0]
    subprocess.run(["xdotool", "windowfocus", window], check=True)
    return window


def clickAt(desktop, window: str, x: float, y: float) -> None:
    subprocess.run(["xdotool", "mousemove", "--window", window, str(round(x)), str(round(y))], check=True)
    pump(desktop)
    subprocess.run(["xdotool", "mousedown", "1"], check=True)
    pump(desktop)
    subprocess.run(["xdotool", "mouseup", "1"], check=True)
    pump(desktop)


def checkTitle(desktop, folder: pathlib.Path, name: str, x: float, y: float) -> None:
    path = folder / f"{name}.png"
    dearpygui.output_frame_buffer(file=str(path))
    pump(desktop, path.is_file)
    with PIL.Image.open(path) as image:
        pixel = image.convert("RGB").getpixel((round(x), round(y)))
    assert all(abs(actual - expected) <= 1 for actual, expected in zip(pixel, theme.colors.selected, strict=True)), (pixel, theme.colors.selected)


def fileDialogs(statePath: pathlib.Path, modelPath: pathlib.Path) -> None:
    workspace = Workspace(statePath)
    workspace.configure(str(statePath / "workspace"), setupComplete=True)
    workspace.setAppearance("Dark", (235, 130, 65))
    desktop = Desktop(workspace)
    original = modelPath.read_bytes()
    try:
        desktop.start()
        desktop.navigate(ModelsView(desktop))
        pump(desktop)
        window = windowId()
        for allFiles in (False, True):
            dearpygui.set_value("modelPathInput", str(modelPath.parent))
            activate("modelPathInputBrowse")
            pump(desktop)
            x = (dearpygui.get_viewport_client_width() - 840) / 2
            y = (dearpygui.get_viewport_client_height() - 510) / 2
            checkTitle(desktop, statePath, f"picker{int(allFiles)}", x + 740, y + 10)
            if allFiles:
                clickAt(desktop, window, x + 820, y + 448)
                clickAt(desktop, window, x + 745, y + 497)
                assert dearpygui.get_file_dialog_info("pathDialog")["current_filter"] == ".*"
            clickAt(desktop, window, x + 275, y + 155)
            clickAt(desktop, window, x + 34, y + 481)
            pump(desktop, lambda: not dearpygui.does_item_exist("pathDialog"))
            assert dearpygui.get_value("modelPathInput") == str(modelPath)
            assert desktop.view.path == str(modelPath)
        activate("registerModelButton")
        pump(desktop, lambda: not desktop.tasks.busy)
        assert workspace.models.records[0].path == str(modelPath)
        assert modelPath.read_bytes() == original
        alternate = modelPath.parent / "typed choice.safetensors"
        alternate.write_bytes(original)
        dearpygui.set_value("modelPathInput", str(modelPath.parent))
        activate("modelPathInputBrowse")
        pump(desktop)
        clickAt(desktop, window, x + 275, y + 155)
        clickAt(desktop, window, x + 400, y + 448)
        subprocess.run(["xdotool", "keydown", "ctrl", "keydown", "a"], check=True)
        pump(desktop)
        subprocess.run(["xdotool", "keyup", "a", "keyup", "ctrl"], check=True)
        pump(desktop)
        keyboard = subprocess.Popen(["xdotool", "type", "--delay", "70", alternate.name])
        pump(desktop, lambda: keyboard.poll() is not None)
        assert keyboard.returncode == 0
        selection = dearpygui.get_file_dialog_info("pathDialog")
        clickAt(desktop, window, x + 34, y + 481)
        assert dearpygui.get_value("modelPathInput") == str(alternate), (dearpygui.get_value("modelPathInput"), selection)
        captionFolder = statePath / "caption files"
        captionFolder.mkdir()
        caption = captionFolder / "caption.v1.txt"
        caption.write_text("A caption shared by several images.", encoding="utf-8")
        dearpygui.set_value("modelPathInput", str(captionFolder))
        desktop.browse("modelPathInput", False, desktop.view.capture, fileExtensions=(".txt",))
        pump(desktop)
        clickAt(desktop, window, x + 820, y + 448)
        clickAt(desktop, window, x + 745, y + 497)
        clickAt(desktop, window, x + 275, y + 155)
        clickAt(desktop, window, x + 34, y + 481)
        assert dearpygui.get_value("modelPathInput") == str(caption)
        assert caption.read_text(encoding="utf-8") == "A caption shared by several images."
        for mode, accent in (("Dark", (215, 75, 145)), ("Light", (45, 125, 190))):
            workspace.setAppearance(mode, accent)
            desktop.applyAppearance()
            desktop.confirm("Theme check", "This confirmation must use the saved accent.", lambda: None, "Confirm")
            pump(desktop)
            x, y = dearpygui.get_item_pos("confirmDialog")
            checkTitle(desktop, statePath, f"confirmation{mode}", x + 480, y + 10)
            activate("cancelConfirmation")
            pump(desktop)
            desktop.showError(ValueError("Dialog color check"))
            for _ in range(5):
                desktop.frame()
            x, y = dearpygui.get_item_pos("errorDialog")
            capture = statePath / f"error{mode}.png"
            dearpygui.output_frame_buffer(file=str(capture))
            for _ in range(5):
                desktop.frame()
            with PIL.Image.open(capture) as image:
                pixel = image.convert("RGB").getpixel((round(x + 530), round(y + 10)))
            assert all(abs(actual - expected) <= 1 for actual, expected in zip(pixel, theme.colors.selected, strict=True))
            desktop.closeDialog("errorDialog")
            pump(desktop)
        assert "torch" not in sys.modules
    finally:
        desktop.close()


def runtimeLocation(statePath: pathlib.Path) -> None:
    workspace = Workspace(statePath)
    workspace.configure(str(statePath / "workspace"), setupComplete=True)
    storage = statePath.parent / "another drive" / "runtime storage"
    storage.mkdir(parents=True)
    desktop = Desktop(workspace)
    try:
        desktop.start()
        desktop.showTraining()
        pump(desktop)
        window = windowId()
        dearpygui.set_value("trainingRuntimePath", str(storage))
        activate("trainingRuntimePathBrowse")
        pump(desktop)
        x = (dearpygui.get_viewport_client_width() - 840) / 2
        y = (dearpygui.get_viewport_client_height() - 510) / 2
        clickAt(desktop, window, x + 34, y + 481)
        pump(desktop, lambda: not dearpygui.does_item_exist("pathDialog"))
        assert pathlib.Path(dearpygui.get_value("trainingRuntimePath")) == storage
        activate("setupTrainingRuntime")
        pump(desktop)
        assert desktop.training.runtime.root == storage / "anima-v1"
        assert Workspace(statePath).preferences.runtimeStoragePath == str(storage)
        assert not desktop.training.runtime.root.exists()
        activate("cancelConfirmation")
        pump(desktop)
        assert not desktop.training.busy
        assert not desktop.training.runtime.root.exists()
        assert not (statePath / "training" / "anima-v1").exists()
        assert str(storage / "anima-v1") in dearpygui.get_value("runtimeInstallPath")
    finally:
        desktop.close()
    reopened = Workspace(statePath)
    assert reopened.preferences.runtimeStoragePath == str(storage)


if __name__ == "__main__":
    {"files": fileDialogs, "location": runtimeLocation}[sys.argv[1]](*map(pathlib.Path, sys.argv[2:]))
