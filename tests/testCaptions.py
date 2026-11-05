import dataclasses
import threading

import pytest

from animaEasy.captions import MAX_CAPTION_FILE_BYTES, MAX_CAPTION_LENGTH, CaptionProfile, checkCaptions, readCaptionFile, resolveCaption
from animaEasy.datasets import inspectDataset
from animaEasy.domain import Project
from animaEasy.errors import ValidationError
from animaEasy.images import listImages, loadPreview
from animaEasy.workspace import Workspace


def testMatchingCaptionsAreReadOnlyAndDecodeBom(sampleDataset):
    path = sampleDataset / "image0.txt"
    path.write_text("portrait, café — モデル", encoding="utf-8-sig")
    original, modifiedAt = path.read_bytes(), path.stat().st_mtime_ns
    document = resolveCaption(sampleDataset, "image0.png", CaptionProfile())
    assert document.text == "portrait, café — モデル"
    assert document.status == "Ready"
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == modifiedAt


def testOneExternalCaptionCanBeAssignedWithoutMatchingNames(sampleDataset, sandboxFolder):
    sharedPath = sandboxFolder / "shared description.txt"
    sharedPath.write_text("shared caption — café", encoding="utf-8")
    for path in sampleDataset.glob("*.txt"):
        path.unlink()
    names = listImages(str(sampleDataset)).names
    profile = CaptionProfile(mode="files").assignFile(names, str(sharedPath))
    original = sharedPath.read_bytes()
    report = inspectDataset(str(sampleDataset), captionProfile=profile)
    assert report.captionMode == "files"
    assert report.captionCount == 3
    assert report.usable
    for name in names:
        assert resolveCaption(sampleDataset, name, profile).text == "shared caption — café"
    assert sharedPath.read_bytes() == original
    assert not list(sampleDataset.glob("*.txt"))


def testInlineSelectedAndAllAssignmentsPreserveOtherImagesAndFiles(sampleDataset):
    files = {path: path.read_bytes() for path in sampleDataset.iterdir()}
    profile = CaptionProfile(mode="inline").assignText(("image0.png",), "first caption")
    selected = profile.assignText(("image1.png",), "selected caption")
    assert selected.textCaptions == {"image0.png": "first caption", "image1.png": "selected caption"}
    assert profile.textCaptions == {"image0.png": "first caption"}
    assert inspectDataset(str(sampleDataset), captionProfile=selected).missingCaptions == 1
    allImages = selected.assignText(listImages(str(sampleDataset)).names, "shared caption")
    assert inspectDataset(str(sampleDataset), captionProfile=allImages).captionCount == 3
    cleared = allImages.assignText(("image1.png",), "")
    assert "image1.png" not in cleared.textCaptions
    assert inspectDataset(str(sampleDataset), captionProfile=cleared).missingCaptions == 1
    assert {path: path.read_bytes() for path in sampleDataset.iterdir()} == files


def testSwitchingCaptionModesPreservesInactiveWork(sampleDataset, sandboxFolder):
    external = sandboxFolder / "caption.txt"
    external.write_text("external description", encoding="utf-8")
    profile = CaptionProfile(mode="inline").assignText(("image0.png",), "project description")
    assigned = dataclasses.replace(profile, mode="files").assignFile(("image0.png",), str(external))
    assert resolveCaption(sampleDataset, "image0.png", assigned).text == "external description"
    matching = dataclasses.replace(assigned, mode="matching")
    assert resolveCaption(sampleDataset, "image0.png", matching).text.startswith("subject, example 0")
    restored = dataclasses.replace(matching, mode="inline")
    assert resolveCaption(sampleDataset, "image0.png", restored).text == "project description"
    assert restored.fileAssignments == assigned.fileAssignments


def testCaptionProfilesStayWithTheirDatasetFolderAfterReopening(sampleDataset, sandboxFolder):
    workspace = Workspace(sandboxFolder / "state")
    project = workspace.projects.create("Caption profiles")
    project = dataclasses.replace(project, datasetPath=str(sampleDataset)).withCaptions(CaptionProfile(mode="inline").assignText(("image0.png",), "original folder"))
    otherFolder = sandboxFolder / "another folder"
    otherFolder.mkdir()
    other = dataclasses.replace(project, datasetPath=str(otherFolder))
    assert other.captions.mode == "matching"
    assert not other.captions.textCaptions
    other = other.withCaptions(CaptionProfile(mode="inline").assignText(("image0.png",), "other folder"))
    workspace.projects.save(other)
    reopened = Workspace(sandboxFolder / "state").projects.load(project.id)
    assert reopened.captions.textCaptions["image0.png"] == "other folder"
    original = dataclasses.replace(reopened, datasetPath=str(sampleDataset / "."))
    assert original.captions.textCaptions["image0.png"] == "original folder"
    assert len(original.captionProfiles) == 2


def testOldProjectRecordsLoadWithoutCaptionFields():
    data = Project.create("Older project").toData()
    data.pop("captionProfiles")
    data["training"].pop("effortPreset")
    restored = Project.fromData(data)
    assert restored.captions == CaptionProfile()
    assert restored.training.effortPreset == "Custom"


@pytest.mark.parametrize("name", ["../outside.png", "folder/image.png", "folder\\image.png", "", ".", "..", "image\x00.png"])
def testAssignmentsRejectPathsInsteadOfImageNames(name):
    with pytest.raises(ValidationError):
        CaptionProfile().assignText((name,), "caption")


def testInlineValidationAndClearingAssignments(sampleDataset):
    with pytest.raises(ValidationError):
        CaptionProfile().assignText(("image0.png",), "x" * (MAX_CAPTION_LENGTH + 1))
    with pytest.raises(ValidationError):
        CaptionProfile().assignText(("image0.png",), "invalid\x00caption")
    assigned = CaptionProfile(mode="files").assignFile(("image0.png",), str(sampleDataset / "image0.txt"))
    cleared = assigned.assignFile(("image0.png",), "")
    assert not cleared.fileAssignments
    assert resolveCaption(sampleDataset, "image0.png", cleared).status == "Unassigned"


@pytest.mark.parametrize("content, expected", [(b" \n", "Empty"), (b"\xff\xfe", "Invalid"), (b"before\x00after", "Invalid"), (b"caption", "Ready")])
def testExternalCaptionValidation(sandboxFolder, content, expected):
    path = sandboxFolder / "caption.txt"
    path.write_bytes(content)
    assert readCaptionFile(str(path)).status == expected
    assert path.read_bytes() == content


def testExternalCaptionSizeLimitAndMissingPath(sandboxFolder):
    path = sandboxFolder / "large.txt"
    path.write_bytes(b"a" * (MAX_CAPTION_FILE_BYTES + 1))
    assert readCaptionFile(str(path)).status == "Invalid"
    assert readCaptionFile(str(sandboxFolder / "missing.txt")).status == "Missing"
    assert readCaptionFile("").status == "Unassigned"


def testCaptionChecksRespectCancellation(sampleDataset):
    stopped = threading.Event()
    stopped.set()
    with pytest.raises(ValidationError, match="canceled"):
        checkCaptions(sampleDataset, ("image0.png",), CaptionProfile(), stopped)


def testImageInventoryAndBoundedPreview(sampleDataset):
    index = listImages(str(sampleDataset))
    assert index.names == ("image0.png", "image1.png", "image2.png")
    assert index.nestedFolders == 0
    preview = loadPreview(sampleDataset / "image0.png")
    assert preview.originalSize == (64, 96)
    assert 0 < preview.width <= 192 and 0 < preview.height <= 144
    assert len(preview.pixels) == preview.width * preview.height * 4
    assert all(0 <= channel <= 1 for channel in preview.pixels)
    assert preview.pixels[3] == 1


def testCorruptImageCannotProduceAMisleadingPreview(sampleDataset):
    path = sampleDataset / "image0.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValidationError, match="Cannot preview"):
        loadPreview(path)


def testRelativeAssignedCaptionPathsUseTheDatasetFolder(sampleDataset):
    profile = CaptionProfile(mode="files").assignFile(("image2.png",), "image0.txt")
    document = resolveCaption(sampleDataset, "image2.png", profile)
    assert document.status == "Ready"
    assert document.text.startswith("subject, example 0")
    assert document.source == str(sampleDataset / "image0.txt")
