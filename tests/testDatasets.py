import threading

import PIL.Image
import pytest

from animaEasy.datasets import inspectDataset
from animaEasy.errors import ValidationError


def testActualImagesAndCaptions(sampleDataset):
    report = inspectDataset(str(sampleDataset))
    assert report.imageCount == 3
    assert report.captionCount == 3
    assert report.usable
    assert not report.issues


def testMissingAndEmptyCaptions(sampleDataset):
    (sampleDataset / "image0.txt").unlink()
    (sampleDataset / "image1.txt").write_text("  \n", encoding="utf-8")
    report = inspectDataset(str(sampleDataset))
    assert report.missingCaptions == 2
    assert report.captionCount == 1
    assert report.usable
    assert len(report.issues) == 2


def testCorruptImageAndNonUtf8Caption(sampleDataset):
    (sampleDataset / "image0.png").write_bytes(b"broken")
    (sampleDataset / "image1.txt").write_bytes(b"\xff\xfe\x00")
    report = inspectDataset(str(sampleDataset))
    assert report.invalidFiles == 2
    assert not report.usable


def testNestedImagesAreNotSilentlyIncluded(sampleDataset):
    nested = sampleDataset / "nested"
    nested.mkdir()
    PIL.Image.new("RGB", (32, 32)).save(nested / "other.png")
    report = inspectDataset(str(sampleDataset))
    assert report.imageCount == 3
    assert report.nestedFolders == 1
    assert any("ignored" in issue for issue in report.issues)


def testDistinctImageExtensionsMayShareACaptionStem(sampleDataset):
    PIL.Image.new("RGB", (32, 32)).save(sampleDataset / "image0.jpg")
    report = inspectDataset(str(sampleDataset))
    assert report.usable
    assert report.imageCount == 4
    assert report.captionCount == 4


def testDatasetCheckCanBeCanceledDuringShutdown(sampleDataset):
    cancelEvent = threading.Event()
    cancelEvent.set()
    with pytest.raises(ValidationError, match="canceled"):
        inspectDataset(str(sampleDataset), cancelEvent)
    assert len(list(sampleDataset.glob("*.png"))) == 3


def testEmptyAndMissingFolders(sandboxFolder):
    assert not inspectDataset(str(sandboxFolder)).usable
    with pytest.raises(ValidationError):
        inspectDataset(str(sandboxFolder / "missing"))
    with pytest.raises(ValidationError):
        inspectDataset("")
