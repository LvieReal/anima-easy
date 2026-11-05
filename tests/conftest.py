import dataclasses
import datetime
import pathlib
import tempfile

import numpy
import PIL.Image
import pytest
import safetensors.numpy

from animaEasy.domain import MODEL_ROLES, TrainingSettings
from animaEasy.models import inspectModel
from animaEasy.training import TrainingController
from animaEasy.trainingRuns import RunStore
from animaEasy.workspace import Workspace


@pytest.fixture
def sandboxFolder():
    date = datetime.datetime.now(datetime.UTC).date().isoformat()
    with tempfile.TemporaryDirectory(prefix=f"animaEasyTests-{date}-") as folder:
        yield pathlib.Path(folder)


@pytest.fixture
def sampleDataset(sandboxFolder):
    folder = sandboxFolder / "dataset with spaces"
    folder.mkdir()
    for index, color in enumerate(((71, 106, 90), (122, 145, 97), (66, 85, 94))):
        PIL.Image.new("RGB", (64, 96), color).save(folder / f"image{index}.png")
        (folder / f"image{index}.txt").write_text(f"subject, example {index}, portrait", encoding="utf-8")
    return folder


@pytest.fixture
def sampleModel(sandboxFolder):
    path = sandboxFolder / "model with spaces.safetensors"
    safetensors.numpy.save_file({"test.weight": numpy.arange(12, dtype=numpy.float32).reshape(3, 4)}, path)
    return path


@pytest.fixture
def trainingInputs(sandboxFolder, sampleDataset):
    workspace = Workspace(sandboxFolder / "state")
    workspace.configure(str(sandboxFolder / "projects with spaces"), setupComplete=True)
    models = {}
    for role in MODEL_ROLES:
        path = sandboxFolder / f"{role} café.safetensors"
        safetensors.numpy.save_file({"fixture.weight": numpy.arange(12, dtype=numpy.float32)}, path)
        models[role] = workspace.models.register(inspectModel(str(path)), role)
    project = workspace.projects.create("Captured training")
    project = workspace.projects.save(dataclasses.replace(project, datasetPath=str(sampleDataset), modelIds={role: record.id for role, record in models.items()}, training=TrainingSettings(epochs=3, repeats=2, resolution=256)))
    controller = TrainingController(workspace)
    controller.stateKeyPath.parent.mkdir(parents=True, exist_ok=True)
    controller.ensureStateKey()
    store = RunStore(workspace.projects, controller.stateKeyPath)
    record = store.create(project, controller.runtime.fingerprint)
    folder = store.folder(project.id, record["id"])
    yield workspace, project, models, controller, store, record, folder
    controller.close()
