import pytest

from animaEasy.errors import StorageError
from animaEasy.storage import readJson, validateFolder, writeJson


def testAtomicRoundTrip(sandboxFolder):
    path = sandboxFolder / "state" / "project.json"
    data = {"schemaVersion": 1, "name": "Café — Модель"}
    writeJson(path, data)
    assert readJson(path) == data
    writeJson(path, {**data, "name": "Updated"})
    assert readJson(path)["name"] == "Updated"
    assert not list(path.parent.glob("*.tmp"))


def testFailedSerializationPreservesPreviousState(sandboxFolder):
    path = sandboxFolder / "state.json"
    writeJson(path, {"schemaVersion": 1, "name": "Keep this"})
    previous = path.read_bytes()
    with pytest.raises(StorageError):
        writeJson(path, {"schemaVersion": 1, "number": float("nan")})
    assert path.read_bytes() == previous


@pytest.mark.parametrize("content", ['{broken', '{"schemaVersion": 2}', '[]', '{"schemaVersion": true}', '{}'])
def testDamagedOrUnsupportedStateIsNotReplaced(sandboxFolder, content):
    path = sandboxFolder / "settings.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(StorageError):
        readJson(path)
    assert path.read_text(encoding="utf-8") == content


def testWorkspacePathMustBeWritableDirectory(sandboxFolder):
    assert validateFolder(str(sandboxFolder / "new workspace")).is_dir()
    filePath = sandboxFolder / "not a folder"
    filePath.write_text("keep", encoding="utf-8")
    with pytest.raises(StorageError):
        validateFolder(str(filePath))
    assert filePath.read_text(encoding="utf-8") == "keep"
