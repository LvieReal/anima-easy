import dataclasses
import pathlib

from animaEasy.appearance import DEFAULT_ACCENT, validateAppearance
from animaEasy.domain import Project, newId
from animaEasy.errors import StorageError, ValidationError
from animaEasy.models import ModelLibrary
from animaEasy.projects import ProjectStore
from animaEasy.storage import readJson, validateFolder, writeJson


@dataclasses.dataclass(frozen=True)
class Preferences:
    workspacePath: str
    setupComplete: bool = False
    theme: str = "Dark"
    accentColor: tuple[int, int, int] = DEFAULT_ACCENT
    runtimeStoragePath: str = ""


class Workspace:
    def __init__(self, statePath: pathlib.Path):
        self.statePath = statePath.expanduser().resolve()
        self.preferencesPath = self.statePath / "settings.json"
        self.preferences = self.loadPreferences()
        self.models = ModelLibrary(self.statePath)
        self.projects = ProjectStore(pathlib.Path(self.preferences.workspacePath))

    def loadPreferences(self) -> Preferences:
        if not self.preferencesPath.exists():
            return Preferences(str(self.statePath / "workspace"))
        data = readJson(self.preferencesPath)
        if not isinstance(data.get("workspacePath"), str) or not data["workspacePath"].strip() or type(data.get("setupComplete")) is not bool:
            raise StorageError("Invalid settings.json. The file has not been replaced.")
        if not pathlib.Path(data["workspacePath"]).is_absolute():
            raise StorageError("The workspace path in settings.json must be absolute.")
        mode, accent = data.get("theme", "Dark"), data.get("accentColor", DEFAULT_ACCENT)
        try:
            validateAppearance(mode, accent)
        except ValidationError as error:
            raise StorageError(f"Invalid appearance settings: {error}. The file has not been replaced.") from error
        runtimeStoragePath = data.get("runtimeStoragePath", "")
        if not isinstance(runtimeStoragePath, str) or "\x00" in runtimeStoragePath or runtimeStoragePath and not pathlib.Path(runtimeStoragePath).is_absolute():
            raise StorageError("The runtime storage path in settings.json must be absolute. The file has not been replaced.")
        return Preferences(data["workspacePath"], data["setupComplete"], mode, tuple(accent), runtimeStoragePath)

    def savePreferences(self, preferences: Preferences) -> None:
        writeJson(self.preferencesPath, {"schemaVersion": 1, **dataclasses.asdict(preferences)})
        self.preferences = preferences

    def setAppearance(self, mode: str, accent: tuple[int, int, int]) -> None:
        validateAppearance(mode, accent)
        self.savePreferences(dataclasses.replace(self.preferences, theme=mode, accentColor=tuple(accent)))

    def setRuntimeStorage(self, path: str) -> pathlib.Path:
        folder = validateFolder(path, "runtime storage")
        self.savePreferences(dataclasses.replace(self.preferences, runtimeStoragePath=str(folder)))
        return folder

    def preserveDraft(self, project: Project) -> pathlib.Path:
        path = self.statePath / "recovery" / f"{project.id}-{newId()}.json"
        writeJson(path, project.toData())
        return path

    def configure(self, workspacePath: str, setupComplete: bool | None = None) -> None:
        folder = validateFolder(workspacePath)
        self.savePreferences(dataclasses.replace(self.preferences, workspacePath=str(folder), setupComplete=self.preferences.setupComplete if setupComplete is None else setupComplete))
        self.projects = ProjectStore(folder)
