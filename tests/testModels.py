import hashlib
import json
import shutil
import subprocess

import numpy
import pytest
import safetensors.numpy

from animaEasy.errors import StorageError, ValidationError
from animaEasy.models import ModelLibrary, inspectModel


def testRegistrationDoesNotCopyOrModifyWeights(sandboxFolder, sampleModel):
    original = hashlib.sha256(sampleModel.read_bytes()).hexdigest()
    library = ModelLibrary(sandboxFolder / "state")
    inspected = inspectModel(str(sampleModel))
    record = library.register(inspected, "diffusion", "Anima test fixture", "Custom Anima model")
    assert record.path == str(sampleModel.resolve())
    assert record.tensorCount == 1
    assert record.fileStatus() == "Registered"
    assert hashlib.sha256(sampleModel.read_bytes()).hexdigest() == original
    assert not list((sandboxFolder / "state").glob("*.safetensors"))
    reopened = ModelLibrary(sandboxFolder / "state")
    assert reopened.get(record.id) == record


def testRepeatedRegistrationUpdatesTheSameEntry(sandboxFolder, sampleModel):
    library = ModelLibrary(sandboxFolder)
    first = library.register(inspectModel(str(sampleModel)), "diffusion")
    second = library.register(inspectModel(str(sampleModel)), "diffusion", "New label")
    assert first.id == second.id
    assert len(library.records) == 1
    assert library.records[0].label == "New label"
    with pytest.raises(ValidationError):
        library.register(inspectModel(str(sampleModel)), "vae")


def testForgetDoesNotDeleteTheOriginal(sandboxFolder, sampleModel):
    library = ModelLibrary(sandboxFolder / "state")
    record = library.register(inspectModel(str(sampleModel)), "textEncoder")
    library.forget(record.id)
    assert not ModelLibrary(sandboxFolder / "state").records
    assert sampleModel.is_file()


def testChangedAndMissingModelsAreReported(sandboxFolder, sampleModel):
    library = ModelLibrary(sandboxFolder)
    record = library.register(inspectModel(str(sampleModel)), "vae")
    safetensors.numpy.save_file({"different.weight": numpy.ones((6, 8), dtype=numpy.float32)}, sampleModel)
    assert record.fileStatus() == "Changed; register again"
    refreshed = library.register(inspectModel(str(sampleModel)), "vae")
    assert refreshed.id == record.id
    assert refreshed.fileStatus() == "Registered"
    sampleModel.unlink()
    assert refreshed.fileStatus() == "Missing file"


def testIncompleteDownloadsAreRejected(sandboxFolder, sampleModel):
    partial = sandboxFolder / "partial.safetensors"
    partial.write_bytes(sampleModel.read_bytes()[:-8])
    with pytest.raises(ValidationError):
        inspectModel(str(partial))


@pytest.mark.parametrize("content", [b"", b"not a model", b"\xff" * 32])
def testInvalidModelFiles(sandboxFolder, content):
    path = sandboxFolder / "invalid.safetensors"
    path.write_bytes(content)
    with pytest.raises(ValidationError):
        inspectModel(str(path))


def testEmptyAndUnsafeFormatsAreRejected(sandboxFolder):
    empty = sandboxFolder / "empty.safetensors"
    safetensors.numpy.save_file({}, empty)
    with pytest.raises(ValidationError):
        inspectModel(str(empty))
    with pytest.raises(ValidationError):
        inspectModel(str(sandboxFolder / "weights.pth"))


def testInspectionRejectsFileChangesBeforeRegistration(sandboxFolder, sampleModel):
    inspection = inspectModel(str(sampleModel))
    sampleModel.unlink()
    with pytest.raises(ValidationError):
        ModelLibrary(sandboxFolder).register(inspection, "diffusion")


def testCorruptLibraryIsPreserved(sandboxFolder):
    path = sandboxFolder / "models.json"
    path.write_text('{"schemaVersion": 1, "models": [42]}', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(StorageError):
        ModelLibrary(sandboxFolder)
    assert path.read_bytes() == before


def testInspectionDoesNotImportTrainingLibraries(sampleModel):
    command = "import json, sys; from animaEasy.models import inspectModel; result = inspectModel(sys.argv[1]); print(json.dumps({'tensors': result.tensorCount, 'loaded': sorted(set(sys.modules) & {'torch', 'gradio'})}))"
    result = subprocess.run([shutil.which("uv"), "run", "--no-sync", "python", "-c", command, str(sampleModel)], capture_output=True, text=True, check=True, timeout=20)
    assert json.loads(result.stdout) == {"tensors": 1, "loaded": []}
