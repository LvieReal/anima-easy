import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys
import time

from dearpygui import dearpygui

from animaEasy.domain import Project
from animaEasy.errors import StorageError
from animaEasy.storage import readJson
from animaEasy.ui.editorView import EditorView
from animaEasy.ui.modelsView import ModelsView
from animaEasy.ui.shell import Desktop
from animaEasy.workspace import Workspace


def pump(desktop, condition=lambda: True, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    frames = 0
    while time.monotonic() < deadline:
        desktop.frame()
        frames += 1
        if frames >= 3 and condition():
            assert not dearpygui.does_item_exist("errorDialog")
            return
        time.sleep(0.01)
    raise AssertionError("The desktop did not reach the expected state.")


def activate(tag: str) -> None:
    options = dearpygui.get_item_configuration(tag)
    callback = options["callback"]
    assert callback is not None
    dearpygui.run_callbacks([[callback, tag, dearpygui.get_value(tag), options.get("user_data")]])


def workflow(statePath: pathlib.Path, datasetPath: pathlib.Path, modelPath: pathlib.Path) -> None:
    workspace = Workspace(statePath)
    desktop = Desktop(workspace)
    try:
        desktop.start()
        pump(desktop)
        activate("saveWorkspaceButton")
        pump(desktop)
        for role, label in (("Diffusion model", "Diffusion fixture"), ("Text encoder", "Encoder fixture"), ("VAE", "VAE fixture")):
            path = statePath.parent / f"{label}.safetensors"
            shutil.copyfile(modelPath, path)
            dearpygui.set_value("modelRoleInput", role)
            activate("modelRoleInput")
            dearpygui.set_value("modelPathInput", str(path))
            dearpygui.set_value("modelLabelInput", label)
            activate("registerModelButton")
            pump(desktop, lambda: not desktop.tasks.busy)
        assert len(workspace.models.records) == 3
        activate("continueSetupButton")
        pump(desktop)
        activate("finishSetupButton")
        pump(desktop)
        activate("newProjectButton")
        pump(desktop)
        dearpygui.set_value("projectNameInput", "Portrait study — Модель")
        activate("createProjectButton")
        pump(desktop)
        assert isinstance(desktop.view, EditorView)
        projectId = desktop.view.project.id
        dearpygui.set_value("datasetPathInput", str(datasetPath))
        activate("datasetPathInput")
        dearpygui.set_value("projectNotesInput", "UTF-8 notes — café")
        activate("projectNotesInput")
        activate("checkDatasetButton")
        pump(desktop, lambda: not desktop.tasks.busy)
        assert desktop.view.report.imageCount == 3
        activate("nextStepButton")
        pump(desktop)
        for role in ("diffusion", "textEncoder", "vae"):
            tag = f"{role}Choice"
            options = dearpygui.get_item_configuration(tag)["items"]
            dearpygui.set_value(tag, options[-1])
            activate(tag)
        activate("nextStepButton")
        pump(desktop)
        dearpygui.set_value("trainingRank", 32)
        activate("trainingRank")
        dearpygui.set_value("trainingAlpha", 32)
        activate("trainingAlpha")
        activate("nextStepButton")
        pump(desktop)
        assert desktop.view.project.currentStep == 3
        desktop.keyboardSave()
        if sys.platform == "linux" and shutil.which("xdotool"):
            result = subprocess.run(["xdotool", "search", "--all", "--pid", str(os.getpid()), "--name", "^Anima Easy$"], capture_output=True, text=True, check=True)
            windowId = result.stdout.splitlines()[0]
            activation = subprocess.run(["xdotool", "windowactivate", "--sync", windowId], capture_output=True, timeout=5)
            if activation.returncode:
                subprocess.run(["xdotool", "windowfocus", "--sync", windowId], check=True, timeout=5)
            pump(desktop)
            try:
                subprocess.run(["xdotool", "keydown", "Control_L"], check=True)
                pump(desktop, lambda: dearpygui.is_key_down(dearpygui.mvKey_LControl), timeout=3)
                desktop.view.project = dataclasses.replace(desktop.view.project, notes="Native shortcut test")
                desktop.view.markChanged()
                subprocess.run(["xdotool", "keydown", "s"], check=True)
                pump(desktop)
                assert desktop.view.dirtyAt is None
            finally:
                subprocess.run(["xdotool", "keyup", "s", "Control_L"], check=True)
            desktop.view.project = dataclasses.replace(desktop.view.project, notes="UTF-8 notes — café")
            desktop.view.markChanged()
        assert "torch" not in sys.modules
        assert "gradio" not in sys.modules
    finally:
        desktop.close()
    project = Workspace(statePath).projects.load(projectId)
    assert project.training.rank == 32
    assert project.training.alpha == 32
    assert project.notes == "UTF-8 notes — café"
    assert len(project.modelIds) == 3


def restart(statePath: pathlib.Path) -> None:
    workspace = Workspace(statePath)
    assert workspace.preferences.setupComplete
    project = workspace.projects.list().projects[0]
    desktop = Desktop(workspace)
    try:
        desktop.start()
        desktop.openProject(project.id)
        pump(desktop)
        assert desktop.view.project.currentStep == 3
        assert desktop.view.project.name == "Portrait study — Модель"
        dearpygui.set_viewport_width(1024)
        dearpygui.set_viewport_height(700)
        pump(desktop)
        position = dearpygui.get_item_rect_min("saveProjectButton")
        size = dearpygui.get_item_rect_size("saveProjectButton")
        assert position[1] + size[1] < dearpygui.get_viewport_client_height()
        desktop.view.selectStep(userData=0)
        dearpygui.set_value("projectNotesInput", "Saved when closing immediately")
        activate("projectNotesInput")
    finally:
        desktop.close()


def importError(statePath: pathlib.Path) -> None:
    desktop = Desktop(Workspace(statePath))
    try:
        desktop.start()
        desktop.navigate(ModelsView(desktop))
        activate("registerModelButton")
        deadline = time.monotonic() + 10
        while desktop.tasks.busy and time.monotonic() < deadline:
            desktop.frame()
            time.sleep(0.01)
        assert dearpygui.does_item_exist("errorDialog")
        assert dearpygui.get_item_configuration("registerModelButton")["enabled"]
        assert not desktop.workspace.models.records
    finally:
        desktop.close()


def shutdownConflict(statePath: pathlib.Path) -> None:
    workspace = Workspace(statePath)
    workspace.configure(str(statePath / "workspace"), setupComplete=True)
    project = workspace.projects.create("Recovery test")
    desktop = Desktop(workspace)
    desktop.start()
    desktop.openProject(project.id)
    pump(desktop)
    dearpygui.set_value("projectNotesInput", "Unsaved local edit")
    activate("projectNotesInput")
    workspace.projects.save(dataclasses.replace(project, notes="Newer external edit"))
    try:
        desktop.close()
    except StorageError as error:
        assert "recovery draft" in str(error)
    else:
        raise AssertionError("A stale shutdown save should not overwrite a newer project.")
    recovered = list((statePath / "recovery").glob("*.json"))
    assert len(recovered) == 1
    assert Project.fromData(readJson(recovered[0])).notes == "Unsaved local edit"
    assert workspace.projects.load(project.id).notes == "Newer external edit"


if __name__ == "__main__":
    scenario = sys.argv[1]
    paths = [pathlib.Path(value) for value in sys.argv[2:]]
    if scenario == "workflow":
        workflow(*paths)
    elif scenario == "restart":
        restart(*paths)
    elif scenario == "error":
        importError(*paths)
    elif scenario == "recovery":
        shutdownConflict(*paths)
    else:
        raise ValueError(f"Unknown desktop scenario: {scenario}")
