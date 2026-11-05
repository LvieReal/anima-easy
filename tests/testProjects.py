import concurrent.futures
import dataclasses

import pytest

from animaEasy.domain import newId
from animaEasy.errors import StorageError, ValidationError
from animaEasy.projects import ProjectStore
from animaEasy.workspace import Workspace


def testProjectPersistenceAndIsolation(sandboxFolder):
    store = ProjectStore(sandboxFolder)
    first = store.create("First project")
    second = store.create("Second project")
    saved = store.save(dataclasses.replace(first, notes="Модель — café", datasetPath="C:/datasets/a folder", training=dataclasses.replace(first.training, rank=32), currentStep=2))
    reopened = ProjectStore(sandboxFolder)
    assert reopened.load(saved.id) == saved
    assert reopened.load(second.id).notes == ""
    assert reopened.list().projects[0].id == first.id
    assert store.folder(first.id) != store.folder(second.id)


def testProjectNamesAreNotUsedAsPaths(sandboxFolder):
    store = ProjectStore(sandboxFolder)
    project = store.create("../../outside")
    assert store.folder(project.id).parent == sandboxFolder / "projects"
    with pytest.raises(ValidationError):
        store.folder("../../outside")


def testExternalChangesAreNotOverwritten(sandboxFolder):
    store = ProjectStore(sandboxFolder)
    original = store.create("Example")
    saved = store.save(dataclasses.replace(original, notes="Newer disk version"))
    with pytest.raises(StorageError):
        store.save(dataclasses.replace(original, notes="Stale edit"))
    assert store.load(original.id) == saved


def testConcurrentWritersCannotLoseChanges(sandboxFolder):
    store = ProjectStore(sandboxFolder)
    project = store.create("Shared project")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        attempts = [executor.submit(store.save, dataclasses.replace(project, notes=note)) for note in ("First writer", "Second writer")]
        saved = []
        rejected = []
        for attempt in attempts:
            try:
                saved.append(attempt.result(timeout=10))
            except StorageError as error:
                rejected.append(error)
    assert len(saved) == len(rejected) == 1
    assert store.load(project.id) == saved[0]


def testBrokenProjectDoesNotHideHealthyProjects(sandboxFolder):
    store = ProjectStore(sandboxFolder)
    healthy = store.create("Healthy")
    broken = store.root / newId()
    broken.mkdir()
    (broken / "project.json").write_text("{broken", encoding="utf-8")
    result = store.list()
    assert [project.id for project in result.projects] == [healthy.id]
    assert len(result.errors) == 1
    assert (broken / "project.json").read_text(encoding="utf-8") == "{broken"


def testProjectSymlinksAreNotFollowed(sandboxFolder):
    store = ProjectStore(sandboxFolder)
    project = store.create("Example")
    link = store.root / newId()
    try:
        link.symlink_to(store.folder(project.id), target_is_directory=True)
    except OSError:
        pytest.skip("This platform does not permit symlink creation for this user.")
    with pytest.raises(StorageError):
        store.load(link.name)


def testWorkspaceSwitchingPreservesProjectsAndModelRegistry(sandboxFolder):
    state = sandboxFolder / "state"
    workspace = Workspace(state)
    workspace.configure(str(sandboxFolder / "first"), setupComplete=True)
    first = workspace.projects.create("First workspace")
    firstPath = workspace.projects.folder(first.id) / "project.json"
    workspace.configure(str(sandboxFolder / "second"))
    assert not workspace.projects.list().projects
    assert firstPath.is_file()
    restored = Workspace(state)
    assert restored.preferences.setupComplete
    assert restored.preferences.workspacePath == str(sandboxFolder / "second")
    restored.configure(str(sandboxFolder / "first"))
    assert restored.projects.load(first.id) == first


def testCorruptPreferencesAreNotReset(sandboxFolder):
    path = sandboxFolder / "settings.json"
    path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(StorageError):
        Workspace(sandboxFolder)
    assert path.read_text(encoding="utf-8") == "{invalid"
