import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys
import threading

from dearpygui import dearpygui

from animaEasy.captions import CAPTION_MODES, MAX_CAPTION_LENGTH
from animaEasy.errors import ValidationError
from animaEasy.ui import theme
from animaEasy.ui.shell import Desktop
from animaEasy.ui.trainingForm import TRAINING_HINTS
from animaEasy.workspace import Workspace
from desktopScenario import activate, pump


def editValue(tag: str, value) -> None:
    dearpygui.set_value(tag, value)
    activate(tag)


def switchCaptionMode(desktop, mode: str) -> None:
    editValue("captionMode", CAPTION_MODES[mode])
    pump(desktop, lambda: not desktop.tasks.busy and desktop.view.captionView.detail.document is not None)


def captions(statePath: pathlib.Path, datasetPath: pathlib.Path) -> None:
    workspace = Workspace(statePath)
    workspace.configure(str(statePath / "workspace"), setupComplete=True)
    project = workspace.projects.create("Caption studio")
    workspace.projects.save(dataclasses.replace(project, datasetPath=str(datasetPath)))
    originals = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in datasetPath.iterdir()}
    shared = statePath.parent / "shared caption with spaces.txt"
    shared.write_text("external caption — café", encoding="utf-8-sig")
    sharedBytes = shared.read_bytes()
    otherFolder = statePath.parent / "other dataset"
    otherFolder.mkdir()
    shutil.copyfile(datasetPath / "image0.png", otherFolder / "image0.png")
    gate = threading.Event()
    desktop = Desktop(workspace)
    try:
        desktop.start()
        desktop.openProject(project.id)
        editor = desktop.view
        pump(desktop, lambda: not desktop.tasks.busy and editor.captionView.detail.document is not None)
        assert dearpygui.get_value("captionMode") == CAPTION_MODES["matching"]
        assert dearpygui.get_value("captionReadOnly").startswith("subject, example 0")
        assert dearpygui.get_item_configuration("captionReadOnly")["readonly"]
        assert dearpygui.does_item_exist("captionPreviewTexture")
        desktop.tasks.submit(lambda: gate.wait(10), lambda result: None, desktop.showError)
        activate(editor.captionView.rowTags["image1.png"][1])
        assert dearpygui.get_value("captionReadOnly") == ""
        assert dearpygui.get_value("activeCaptionStatus") == "Reading..."
        gate.set()
        pump(desktop, lambda: not desktop.tasks.busy)
        assert dearpygui.get_value("captionReadOnly").startswith("subject, example 1")
        activate(editor.captionView.rowTags["image0.png"][1])
        pump(desktop, lambda: not desktop.tasks.busy)
        gate.clear()
        left = dearpygui.get_item_pos("sidebar")[0]
        right = left + dearpygui.get_item_rect_size("sidebar")[0]
        buttonLeft = dearpygui.get_item_rect_min("navProjects")[0]
        buttonRight = buttonLeft + dearpygui.get_item_rect_size("navProjects")[0]
        assert buttonLeft >= left + 14 and buttonRight <= right - 14
        switchCaptionMode(desktop, "inline")
        editValue("inlineCaptionInput", "individual caption — café")
        editValue(editor.captionView.rowTags["image1.png"][0], True)
        activate("applyCaptionSelected")
        assert dearpygui.does_item_exist("confirmDialog")
        assert "image1.png" not in editor.project.captions.textCaptions
        activate("cancelConfirmation")
        assert "image1.png" not in editor.project.captions.textCaptions
        activate("applyCaptionSelected")
        activate("acceptConfirmation")
        pump(desktop, lambda: not desktop.tasks.busy and editor.captionView.detail.document is not None)
        assert editor.project.captions.textCaptions["image1.png"] == "individual caption — café"
        assert "image2.png" not in editor.project.captions.textCaptions
        activate(editor.captionView.rowTags["image2.png"][1])
        editValue("inlineCaptionInput", "shared project caption")
        activate("applyCaptionAll")
        activate("acceptConfirmation")
        pump(desktop, lambda: not desktop.tasks.busy and editor.captionView.detail.document is not None)
        assert set(editor.project.captions.textCaptions.values()) == {"shared project caption"}
        activate(editor.captionView.rowTags["image0.png"][1])
        editValue("inlineCaptionInput", "individual survives mode changes")
        switchCaptionMode(desktop, "files")
        activate("assignedCaptionPathBrowse")
        pump(desktop)
        assert dearpygui.does_item_exist("pathDialog")
        callback = dearpygui.get_item_configuration("pathDialog")["callback"]
        dearpygui.run_callbacks([[callback, "pathDialog", {"file_path_name": str(shared)}, None]])
        pump(desktop, lambda: editor.captionView.detail.document is not None and not desktop.tasks.busy)
        assert dearpygui.get_value("captionReadOnly") == "external caption — café"
        activate("applyCaptionAll")
        activate("acceptConfirmation")
        pump(desktop, lambda: editor.captionView.detail.document is not None and not desktop.tasks.busy)
        assert set(editor.project.captions.fileAssignments.values()) == {str(shared)}
        activate("checkDatasetButton")
        pump(desktop, lambda: editor.report is not None and not desktop.tasks.busy)
        assert editor.report.captionMode == "files" and editor.report.captionCount == 3
        switchCaptionMode(desktop, "inline")
        assert dearpygui.get_value("inlineCaptionInput") == "individual survives mode changes"
        assert editor.project.captions.textCaptions["image2.png"] == "shared project caption"
        oldCallback = dearpygui.get_item_configuration("inlineCaptionInput")
        activate(editor.captionView.rowTags["image1.png"][1])
        dearpygui.run_callbacks([[oldCallback["callback"], "inlineCaptionInput", "late image zero edit", oldCallback["user_data"]]])
        assert editor.project.captions.textCaptions["image0.png"] == "late image zero edit"
        assert editor.project.captions.textCaptions["image1.png"] == "shared project caption"
        assert dearpygui.get_value("inlineCaptionInput") == "shared project caption"
        try:
            dearpygui.run_callbacks([[oldCallback["callback"], "inlineCaptionInput", "x" * (MAX_CAPTION_LENGTH + 1), oldCallback["user_data"]]])
        except ValidationError:
            pass
        else:
            raise AssertionError("An oversized caption should be rejected.")
        assert dearpygui.get_value("inlineCaptionInput") == "shared project caption"
        desktop.tasks.submit(lambda: gate.wait(10), lambda result: None, desktop.showError)
        editor.scan()
        editValue("inlineCaptionInput", "new caption while the check is queued")
        gate.set()
        pump(desktop, lambda: not desktop.tasks.busy)
        assert editor.report is None
        assert dearpygui.get_item_configuration("checkDatasetButton")["enabled"]
        editValue("datasetPathInput", str(otherFolder))
        pump(desktop, lambda: editor.captionView.index is not None and not desktop.tasks.busy)
        assert editor.project.captions.mode == "matching"
        assert not editor.project.captions.textCaptions
        switchCaptionMode(desktop, "inline")
        assert dearpygui.get_value("inlineCaptionInput") == ""
        editValue("inlineCaptionInput", "only for the other folder")
        editValue("datasetPathInput", str(datasetPath))
        pump(desktop, lambda: editor.captionView.index is not None and not desktop.tasks.busy)
        assert editor.project.captions.mode == "inline"
        assert dearpygui.get_value("inlineCaptionInput") == "late image zero edit"
        editValue("captionFilter", "image1")
        assert editor.captionView.visibleNames == ["image1.png"]
        activate("applyCaptionAll")
        activate("acceptConfirmation")
        pump(desktop, lambda: not desktop.tasks.busy)
        assert len(editor.project.captions.textCaptions) == 3
        assert set(editor.project.captions.textCaptions.values()) == {"late image zero edit"}
        activate("saveProjectButton")
        reopened = Workspace(statePath).projects.load(project.id)
        assert len(reopened.captionProfiles) == 2
        assert reopened.captions.fileAssignments == editor.project.captions.fileAssignments
        assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in datasetPath.iterdir()} == originals
        assert shared.read_bytes() == sharedBytes
        dearpygui.set_viewport_width(1024)
        dearpygui.set_viewport_height(700)
        pump(desktop)
        assert dearpygui.get_item_rect_min("saveProjectButton")[1] + dearpygui.get_item_rect_size("saveProjectButton")[1] < dearpygui.get_viewport_client_height()
        right = sum(dearpygui.get_item_pos(tag)[0] for tag in ("pageHost", "projectBody", "captionDetail")) + dearpygui.get_item_rect_size("captionDetail")[0]
        assert right <= dearpygui.get_viewport_client_width()
        assert dearpygui.get_item_rect_min("stepButton3")[0] + dearpygui.get_item_rect_size("stepButton3")[0] <= dearpygui.get_viewport_client_width()
    finally:
        gate.set()
        desktop.close()


def reopenCaptions(statePath: pathlib.Path) -> None:
    workspace = Workspace(statePath)
    project = workspace.projects.list().projects[0]
    desktop = Desktop(workspace)
    try:
        desktop.start()
        desktop.openProject(project.id)
        pump(desktop, lambda: desktop.view.captionView.detail.document is not None and not desktop.tasks.busy)
        assert dearpygui.get_value("captionMode") == CAPTION_MODES["inline"]
        assert dearpygui.get_value("inlineCaptionInput") == "late image zero edit"
        switchCaptionMode(desktop, "files")
        assert dearpygui.get_value("captionReadOnly") == "external caption — café"
        assert len(desktop.view.project.captions.fileAssignments) == 3
        switchCaptionMode(desktop, "matching")
        assert dearpygui.get_value("captionReadOnly").startswith("subject, example 0")
        switchCaptionMode(desktop, "inline")
        assert dearpygui.get_value("inlineCaptionInput") == "late image zero edit"
    finally:
        desktop.close()


def appearance(statePath: pathlib.Path) -> None:
    workspace = Workspace(statePath)
    workspace.configure(str(statePath / "workspace"), setupComplete=True)
    project = workspace.projects.create("Effort settings")
    desktop = Desktop(workspace)
    try:
        desktop.start()
        activate("navWorkspace")
        pump(desktop)
        assert dearpygui.get_item_rect_min("themePreference")[1] < dearpygui.get_item_rect_min("accentPreference")[1]
        editValue("themePreference", "Light")
        pump(desktop)
        assert desktop.resolvedTheme == "Light"
        assert theme.colors.background[0] >= 240
        assert dearpygui.get_value(theme.colorTag("appTheme", True, dearpygui.mvThemeCol_WindowBg))[:3] == list(theme.colors.background)
        editValue("accentPreference", "Violet")
        editValue("accentColorInput", (40, 110, 190, 255))
        assert workspace.preferences.accentColor == (40, 110, 190)
        editValue("themePreference", "Auto")
        desktop.nextThemeCheck = 0
        pump(desktop, lambda: not desktop.tasks.busy)
        assert workspace.preferences.theme == "Auto"
        assert desktop.resolvedTheme in ("Light", "Dark")
        editValue("themePreference", "Dark")
        pump(desktop)
        assert theme.colors.background[0] < 40
        desktop.openProject(project.id)
        desktop.view.selectStep(userData=2)
        pump(desktop)
        assert dearpygui.get_value("trainingPreset") == "Custom"
        editValue("trainingPreset", "Low")
        assert (dearpygui.get_value("trainingEpochs"), dearpygui.get_value("trainingRepeats")) == (5, 5)
        editValue("trainingRank", 64)
        assert dearpygui.get_value("trainingPreset") == "Low"
        editValue("trainingEpochs", 7)
        assert dearpygui.get_value("trainingPreset") == "Custom"
        editValue("trainingPreset", "High")
        assert (dearpygui.get_value("trainingEpochs"), dearpygui.get_value("trainingRepeats")) == (20, 10)
        assert dearpygui.get_value("trainingRank") == 64
        editValue("trainingPreset", "Custom")
        assert dearpygui.get_value("trainingEpochs") == 20
        for name in TRAINING_HINTS:
            tag = f"hint{name[0].upper()}{name[1:]}Tooltip"
            assert dearpygui.does_item_exist(tag)
            texts = [dearpygui.get_value(child) for child in dearpygui.get_item_children(tag, 1)]
            assert any(isinstance(text, str) and len(text) > 60 for text in texts)
        if sys.platform == "linux" and shutil.which("xdotool"):
            result = subprocess.run(["xdotool", "search", "--all", "--pid", str(os.getpid()), "--name", "^Anima Easy$"], capture_output=True, text=True, check=True)
            window = result.stdout.splitlines()[0]
            position = dearpygui.get_item_rect_min("hintEffortPreset")
            subprocess.run(["xdotool", "mousemove", "--window", window, str(round(position[0] + 6)), str(round(position[1] + 6))], check=True)
            pump(desktop, lambda: dearpygui.get_item_state("hintEffortPreset")["hovered"] and dearpygui.get_item_state("hintEffortPresetTooltip")["visible"])
        activate("saveProjectButton")
        reopened = Workspace(statePath)
        assert reopened.preferences.theme == "Dark"
        assert reopened.preferences.accentColor == (40, 110, 190)
        assert reopened.projects.load(project.id).training.rank == 64
    finally:
        desktop.close()


def pagination(statePath: pathlib.Path, datasetPath: pathlib.Path) -> None:
    workspace = Workspace(statePath)
    workspace.configure(str(statePath / "workspace"), setupComplete=True)
    project = workspace.projects.create("Many images")
    workspace.projects.save(dataclasses.replace(project, datasetPath=str(datasetPath)))
    desktop = Desktop(workspace)
    try:
        desktop.start()
        desktop.openProject(project.id)
        pump(desktop, lambda: desktop.view.captionView.index is not None and not desktop.tasks.busy)
        view = desktop.view.captionView
        assert len(view.visibleNames) == 100
        activate("selectCaptionPage")
        assert len(view.selected) == 100
        activate("captionNextPage")
        assert len(view.visibleNames) == 3
        activate("selectCaptionPage")
        assert len(view.selected) == 103
        editValue("captionFilter", "image102")
        assert view.visibleNames == ["image102.png"]
        assert len(view.selected) == 103
        activate("clearCaptionSelection")
        assert not view.selected
    finally:
        desktop.close()


if __name__ == "__main__":
    scenarios = {"captions": captions, "reopen": reopenCaptions, "appearance": appearance, "pagination": pagination}
    scenarios[sys.argv[1]](*(pathlib.Path(value) for value in sys.argv[2:]))
