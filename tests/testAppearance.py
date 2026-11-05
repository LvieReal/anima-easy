import dataclasses

import pytest

from animaEasy.appearance import ACCENT_OPTIONS, DEFAULT_ACCENT, contrast, resolveTheme
from animaEasy.errors import StorageError, ValidationError
from animaEasy.storage import readJson, writeJson
from animaEasy.ui.theme import makeColors
from animaEasy.workspace import Workspace


def testAppearanceIsSavedAndWorkspaceSwitchDoesNotResetIt(sandboxFolder):
    workspace = Workspace(sandboxFolder / "state")
    workspace.setAppearance("Light", (40, 110, 190))
    workspace.configure(str(sandboxFolder / "another workspace"), setupComplete=True)
    reopened = Workspace(sandboxFolder / "state")
    assert reopened.preferences.theme == "Light"
    assert reopened.preferences.accentColor == (40, 110, 190)
    assert reopened.preferences.setupComplete
    assert readJson(reopened.preferencesPath)["accentColor"] == [40, 110, 190]


def testOlderPreferencesKeepTheirWorkspaceAndUseNewDefaults(sandboxFolder):
    state = sandboxFolder / "state"
    folder = sandboxFolder / "projects"
    writeJson(state / "settings.json", {"schemaVersion": 1, "workspacePath": str(folder), "setupComplete": True})
    workspace = Workspace(state)
    assert workspace.preferences.workspacePath == str(folder)
    assert workspace.preferences.theme == "Dark"
    assert workspace.preferences.accentColor == DEFAULT_ACCENT


@pytest.mark.parametrize("mode, accent", [("Unknown", DEFAULT_ACCENT), ("Dark", (256, 0, 0)), ("Dark", (True, 0, 0)), ("Light", (0, 1))])
def testInvalidAppearanceDoesNotOverwriteExistingPreferences(sandboxFolder, mode, accent):
    workspace = Workspace(sandboxFolder / "state")
    workspace.setAppearance("Dark", DEFAULT_ACCENT)
    before = workspace.preferencesPath.read_bytes()
    with pytest.raises(ValidationError):
        workspace.setAppearance(mode, accent)
    assert workspace.preferencesPath.read_bytes() == before


def testCorruptAppearanceIsReportedRatherThanReset(sandboxFolder):
    workspace = Workspace(sandboxFolder / "state")
    values = dataclasses.asdict(workspace.preferences)
    values["accentColor"] = [500, 0, 0]
    writeJson(workspace.preferencesPath, {"schemaVersion": 1, **values})
    before = workspace.preferencesPath.read_bytes()
    with pytest.raises(StorageError, match="appearance"):
        Workspace(sandboxFolder / "state")
    assert workspace.preferencesPath.read_bytes() == before


@pytest.mark.parametrize("mode", ["Light", "Dark"])
@pytest.mark.parametrize("accent", [*ACCENT_OPTIONS.values(), (0, 0, 0), (255, 255, 255), (255, 255, 0), (118, 118, 118), (122, 122, 122)])
def testThemeAccentAndPrimaryActionRemainReadable(mode, accent):
    colors = makeColors(mode, accent)
    assert contrast(colors.text, colors.background) >= 4.5
    assert contrast(colors.accent, colors.background) >= 4.5
    assert contrast(colors.accent, colors.selected) >= 4.5
    assert contrast(colors.primaryText, colors.rawAccent) >= 4.5
    assert colors.rawAccent == accent


def testExplicitModesAndActualSystemDetection():
    assert resolveTheme("Light") == "Light"
    assert resolveTheme("Dark") == "Dark"
    assert resolveTheme("Auto") in ("Light", "Dark")
    with pytest.raises(ValidationError):
        resolveTheme("other")
