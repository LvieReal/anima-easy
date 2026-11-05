import dataclasses
import datetime
import pathlib

import filelock

from animaEasy.domain import Project, timestamp, validateId
from animaEasy.errors import AnimaError, StorageError
from animaEasy.storage import readJson, writeJson


@dataclasses.dataclass(frozen=True)
class ProjectListing:
    projects: list[Project]
    errors: list[str]


class ProjectStore:
    def __init__(self, workspacePath: pathlib.Path):
        self.root = workspacePath / "projects"

    def folder(self, projectId: str) -> pathlib.Path:
        folder = self.root / validateId(projectId)
        if folder.is_symlink() or folder.is_junction():
            raise StorageError("Project folders cannot be symbolic links.")
        return folder

    def load(self, projectId: str) -> Project:
        project = Project.fromData(readJson(self.folder(projectId) / "project.json"))
        if project.id != projectId:
            raise StorageError(f"Project ID does not match its folder: {projectId}")
        return project

    def create(self, name: str) -> Project:
        project = Project.create(name)
        folder = self.folder(project.id)
        folder.mkdir(parents=True, exist_ok=False)
        try:
            writeJson(folder / "project.json", project.toData())
        except AnimaError:
            folder.rmdir()
            raise
        return project

    def save(self, project: Project) -> Project:
        project.validate()
        with filelock.FileLock(self.folder(project.id) / "project.lock", timeout=5):
            previous = self.load(project.id)
            if previous.updatedAt != project.updatedAt:
                raise StorageError("This project changed on disk. Reopen it before saving so newer changes are not overwritten.")
            saved = dataclasses.replace(project, updatedAt=timestamp())
            writeJson(self.folder(project.id) / "project.json", saved.toData())
            return saved

    def list(self) -> ProjectListing:
        projects = []
        errors = []
        if not self.root.exists():
            return ProjectListing(projects, errors)
        try:
            folders = list(self.root.iterdir())
        except OSError as error:
            raise StorageError(f"Cannot list projects in {self.root}: {error}") from error
        for folder in folders:
            if not folder.is_dir() or not (folder / "project.json").exists():
                continue
            try:
                projects.append(self.load(folder.name))
            except (AnimaError, OSError) as error:
                errors.append(f"{folder.name}: {error}")
        projects.sort(key=lambda project: datetime.datetime.fromisoformat(project.updatedAt), reverse=True)
        return ProjectListing(projects, errors)
