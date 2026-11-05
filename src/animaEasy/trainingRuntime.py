import hashlib
import os
import pathlib
import platform
import re
import shutil
import stat
import tempfile
import threading
import tomllib
import urllib.request
import zipfile

from animaEasy import VERSION
from animaEasy.errors import ValidationError
from animaEasy.trainingAssets import runtimeContract
from animaEasy.trainingProcess import ProcessRunner, trainingEnvironment


TRAINER_REVISION = "cb70b96858477e716a2096292ab8aab433277f38"
TRAINER_ARCHIVE_HASH = "13587ab358c5ecb878a991a391ca924cb37ff1d427805b1476f41bbf23fcc493"
TRAINER_URL = f"https://codeload.github.com/kohya-ss/sd-scripts/zip/{TRAINER_REVISION}"
RUNTIME_ID = "anima-v1"
RUNTIME_PYTHON = "3.12.16"
MINIMUM_UV_VERSION = (0, 13, 7)
ASSET_FOLDER = pathlib.Path(__file__).with_name("trainingAssets")
RUNTIME_FILES = {"runtimeProject.toml": "pyproject.toml", "runtime.lock": "uv.lock", "runtimeProbe.py": "runtimeProbe.py", "runtimeBridge.py": "runtimeBridge.py", "runtimeContract.py": "runtimeContract.py"}


def findUv() -> str:
    executable = "uv.exe" if os.name == "nt" else "uv"
    candidates = [shutil.which(executable), pathlib.Path.home() / ".local" / "bin" / executable, pathlib.Path.home() / ".cargo" / "bin" / executable]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).is_file():
            return str(candidate)
    raise ValidationError("uv was not found. Install uv and reopen Anima Easy using its launcher.")


def runtimeFingerprint() -> str:
    digest = hashlib.sha256(f"{RUNTIME_ID}:{TRAINER_REVISION}:{TRAINER_ARCHIVE_HASH}:{RUNTIME_PYTHON}".encode())
    for name in RUNTIME_FILES:
        digest.update((ASSET_FOLDER / name).read_bytes().replace(b"\r\n", b"\n"))
    digest.update(pathlib.Path(__file__).with_name("trainingArguments.py").read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def extractTrainer(archive: pathlib.Path, destination: pathlib.Path) -> None:
    rootName = f"sd-scripts-{TRAINER_REVISION}"
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if len(entries) > 10000 or sum(item.file_size for item in entries) > 200 * 1024 * 1024:
            raise ValidationError("The trainer source archive exceeds its safety limits.")
        for item in entries:
            relative = pathlib.PurePosixPath(item.filename)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != rootName or "\\" in item.filename or stat.S_ISLNK(item.external_attr >> 16):
                raise ValidationError("The trainer archive contains an unsafe path.")
            if len(relative.parts) == 1:
                continue
            path = destination.joinpath(*relative.parts[1:])
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as inputStream, path.open("wb") as outputStream:
                    shutil.copyfileobj(inputStream, outputStream)


class RuntimeManager:
    def __init__(self, statePath: pathlib.Path, storagePath: pathlib.Path | None = None):
        storage = pathlib.Path(storagePath).expanduser().resolve() if storagePath is not None else statePath / "training"
        self.root = storage / RUNTIME_ID
        self.lockPath = storage / f".{RUNTIME_ID}.lock"
        self.ownershipPath = self.root / "managedRuntime.json"
        self.trainerPath = self.root / "trainer"
        self.logPath = self.root / "setup.log"
        self.readyPath = self.root / "installed.json"
        self.probePath = self.root / "probe.json"
        self.fingerprint = runtimeFingerprint()

    def installed(self) -> bool:
        try:
            record = runtimeContract.readData(self.readyPath)
            pythonPath = self.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            return record.get("fingerprint") == self.fingerprint and pythonPath.is_file() and (self.trainerPath / "anima_train_network.py").is_file() and all((self.root / name).is_file() for name in ("runtimeProbe.py", "runtimeBridge.py", "runtimeContract.py", "uv.lock", "pyproject.toml"))
        except (OSError, ValueError):
            return False

    def probeResult(self) -> dict:
        try:
            return runtimeContract.readData(self.probePath) if self.installed() else {}
        except (OSError, ValueError):
            return {}

    def validateStorage(self) -> None:
        paths = (self.root, self.trainerPath, self.root / ".venv", self.ownershipPath, self.readyPath, self.logPath, self.probePath, self.root / "trainer.download", self.root / "trainer.zip", self.root / "trainer.pending", *[self.root / name for name in RUNTIME_FILES.values()], *self.processFolders())
        if any(runtimeContract.isLink(path) for path in paths):
            raise ValidationError("Managed runtime paths cannot be symbolic links or junctions.")
        if not self.root.exists():
            return
        if not self.root.is_dir():
            raise ValidationError(f"The runtime destination is not a folder: {self.root}. Choose another storage folder.")
        if self.ownershipPath.exists():
            ownership = runtimeContract.readData(self.ownershipPath)
            if ownership.get("schemaVersion") != 1 or ownership.get("runtimeId") != RUNTIME_ID:
                raise ValidationError("This folder has an invalid runtime ownership marker. Choose another storage folder; existing files have not been changed.")
            return
        owned = False
        for evidence in (self.readyPath, self.root / "pyproject.toml", self.root / "trainer.zip"):
            try:
                if not evidence.is_file() or evidence.stat().st_size > 30 * 1024 * 1024:
                    continue
                if evidence == self.readyPath:
                    owned = runtimeContract.readData(evidence).get("runtimeId") == RUNTIME_ID
                elif evidence.suffix == ".toml":
                    project = tomllib.loads(evidence.read_text(encoding="utf-8")).get("project", {})
                    owned = isinstance(project, dict) and project.get("name") == "anima-easy-training-runtime"
                else:
                    owned = runtimeContract.fileDigest(evidence) == TRAINER_ARCHIVE_HASH
            except (OSError, ValueError, TypeError):
                continue
            if owned:
                break
        entries = {path.name for path in self.root.iterdir()}
        if not owned and entries - {"setup.log", "trainer.download"}:
            raise ValidationError(f"The destination already contains files that do not identify an Anima Easy runtime: {self.root}. Choose another storage folder; nothing has been moved or deleted.")

    def claimStorage(self) -> None:
        self.validateStorage()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.ownershipPath.exists():
            runtimeContract.writeData(self.ownershipPath, {"schemaVersion": 1, "runtimeId": RUNTIME_ID})

    def processFolders(self) -> tuple[pathlib.Path, ...]:
        return tuple(self.root / name for name in ("uvCache", "python", "temp", "cache", "huggingface", "torchCache", "torchExtensions", "torchCompileCache", "tritonCache", "cudaCache"))

    def prepareProcessFolders(self) -> None:
        for folder in self.processFolders():
            if runtimeContract.isLink(folder):
                raise ValidationError("Runtime cache and Python directories cannot be symbolic links or junctions.")
            folder.mkdir(parents=True, exist_ok=True)

    def environment(self) -> dict[str, str]:
        environment = {name: value for name, value in trainingEnvironment().items() if not name.startswith("UV_")}
        environment.update({"UV_PROJECT_ENVIRONMENT": str(self.root / ".venv"), "UV_CACHE_DIR": str(self.root / "uvCache"), "UV_PYTHON_INSTALL_DIR": str(self.root / "python"), "UV_PYTHON_INSTALL_BIN": "0", "UV_PYTHON_INSTALL_REGISTRY": "0", "UV_PYTHON_DOWNLOADS": "automatic", "UV_NO_PROGRESS": "1", "XDG_CACHE_HOME": str(self.root / "cache"), "HF_HOME": str(self.root / "huggingface"), "HF_HUB_CACHE": str(self.root / "huggingface" / "hub"), "HUGGINGFACE_HUB_CACHE": str(self.root / "huggingface" / "hub"), "HF_ASSETS_CACHE": str(self.root / "huggingface" / "assets"), "HF_DATASETS_CACHE": str(self.root / "huggingface" / "datasets"), "TORCH_HOME": str(self.root / "torchCache"), "TORCH_EXTENSIONS_DIR": str(self.root / "torchExtensions"), "TORCHINDUCTOR_CACHE_DIR": str(self.root / "torchCompileCache"), "TRITON_CACHE_DIR": str(self.root / "tritonCache"), "CUDA_CACHE_PATH": str(self.root / "cudaCache"), "TMPDIR": str(self.root / "temp"), "TMP": str(self.root / "temp"), "TEMP": str(self.root / "temp")})
        return environment

    def command(self, script: pathlib.Path, *arguments: str) -> list[str]:
        return [findUv(), "run", "--locked", "--no-sync", "--no-python-downloads", "--project", str(self.root), "python", "-I", "-u", str(script), *arguments]

    def downloadSource(self, cancel: threading.Event, onLine) -> pathlib.Path:
        archive = self.root / "trainer.zip"
        if archive.is_file() and runtimeContract.fileDigest(archive, cancel) == TRAINER_ARCHIVE_HASH:
            onLine("Using the verified, cached trainer source archive.")
            return archive
        partial = self.root / "trainer.download"
        request = urllib.request.Request(TRAINER_URL, headers={"User-Agent": f"AnimaEasy/{VERSION}"})
        onLine(f"Downloading pinned trainer source: {TRAINER_REVISION}")
        try:
            with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as stream:
                received = 0
                while chunk := response.read(256 * 1024):
                    if cancel.is_set():
                        raise InterruptedError("Trainer download canceled.")
                    received += len(chunk)
                    if received > 30 * 1024 * 1024:
                        raise ValidationError("The trainer download exceeds its expected size limit.")
                    stream.write(chunk)
            if runtimeContract.fileDigest(partial, cancel) != TRAINER_ARCHIVE_HASH:
                raise ValidationError("The trainer archive failed SHA-256 verification. Nothing was installed from it.")
            os.replace(partial, archive)
        finally:
            partial.unlink(missing_ok=True)
        return archive

    def checkUv(self, runner: ProcessRunner, cancel: threading.Event, onLine) -> str:
        executable = findUv()
        output = []
        def received(line: str) -> None:
            output.append(line)
            onLine(line)
        onLine(f"Checking the runtime installer: {executable}")
        code = runner.run([executable, "--version"], self.root, self.environment(), self.logPath, cancel, received, timeoutSeconds=15)
        if cancel.is_set():
            raise InterruptedError("Runtime setup canceled.")
        match = re.search(r"^uv (\d+)\.(\d+)\.(\d+)", "\n".join(output), re.MULTILINE)
        minimum = ".".join(map(str, MINIMUM_UV_VERSION))
        if code != 0 or match is None:
            raise ValidationError(f"Could not check uv at {executable}. Install uv {minimum} or newer, reopen Anima Easy, and retry setup. See {self.logPath}.")
        if tuple(map(int, match.groups())) < MINIMUM_UV_VERSION:
            detected = ".".join(match.groups())
            raise ValidationError(f"uv {detected} is too old for this managed setup; use uv {minimum} or newer. Older uv releases may not know how to download Python {RUNTIME_PYTHON}. Run uv self update, or update uv with the package manager used to install it. Reopen Anima Easy and retry setup. Python does not need to be installed manually. Installer: {executable}")
        return executable

    def setupMessage(self, message: str, onLine) -> None:
        with self.logPath.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[Anima Easy] {message}\n")
        onLine(message)

    def supportsHardlinks(self) -> bool:
        with tempfile.TemporaryDirectory(prefix="link-check-", dir=self.root / "temp") as folder:
            source = pathlib.Path(folder) / "source"
            source.write_bytes(b"Anima Easy storage check")
            try:
                os.link(source, pathlib.Path(folder) / "link")
            except OSError:
                return False
        return True

    def preparePython(self, executable: str, runner: ProcessRunner, cancel: threading.Event, onLine) -> pathlib.Path:
        output = []
        def received(line: str) -> None:
            output.append(line)
            onLine(line)
        onLine(f"Preparing managed Python {RUNTIME_PYTHON} in {self.root / 'python'}…")
        command = [executable, "python", "install", "--no-bin", "--no-registry", RUNTIME_PYTHON]
        code = runner.run(command, self.root, self.environment(), self.logPath, cancel, received)
        if cancel.is_set():
            raise InterruptedError("Python setup canceled.")
        linkFailure = code != 0 and any("Failed to create Python minor version link directory" in line for line in output)
        if code != 0 and not linkFailure:
            raise ValidationError(f"Managed Python installation failed (exit {code}). See {self.logPath}.")
        output.clear()
        command = [executable, "python", "find", "--system", "--managed-python", "--no-python-downloads", RUNTIME_PYTHON]
        code = runner.run(command, self.root, self.environment(), self.logPath, cancel, received, timeoutSeconds=30)
        if cancel.is_set():
            raise InterruptedError("Python setup canceled.")
        if code != 0 or not output:
            raise ValidationError(f"The exact managed Python {RUNTIME_PYTHON} could not be verified after installation. See {self.logPath}.")
        try:
            python = pathlib.Path(output[-1]).resolve(strict=True)
        except (OSError, ValueError) as error:
            raise ValidationError(f"The managed Python executable is unavailable: {error}. See {self.logPath}.") from error
        if not python.is_file() or not python.is_relative_to((self.root / "python").resolve()):
            raise ValidationError("Managed Python must remain inside the selected runtime storage. No system Python will be substituted.")
        if linkFailure:
            self.setupMessage("uv could not create its optional Python minor-version link. The versioned Python executable was verified; continuing directly without that link or moving Python to another drive.", onLine)
        return python

    def syncDependencies(self, executable: str, python: pathlib.Path, copyPackages: bool, runner: ProcessRunner, cancel: threading.Event, onLine) -> None:
        onLine("Installing locked training dependencies in the isolated environment. Existing model weights are not downloaded or copied.")
        command = [executable, "sync", "--locked", "--no-dev", "--no-build", "--no-python-downloads", "--python", str(python), "--project", str(self.root)]
        if copyPackages:
            command.extend(["--link-mode", "copy"])
        code = runner.run(command, self.root, self.environment(), self.logPath, cancel, onLine)
        if cancel.is_set():
            raise InterruptedError("Runtime setup canceled; run setup again to repair the partial installation.")
        if code != 0:
            raise ValidationError(f"Dependency installation failed (exit {code}). See {self.logPath}.")

    def install(self, runner: ProcessRunner, cancel: threading.Event, onLine) -> dict:
        if platform.system() not in ("Windows", "Linux") or platform.machine().lower() not in ("amd64", "x86_64"):
            raise ValidationError("Managed training supports 64-bit Windows and Linux on x86-64 CPUs.")
        self.claimStorage()
        self.prepareProcessFolders()
        executable = self.checkUv(runner, cancel, onLine)
        copyPackages = not self.supportsHardlinks()
        minimumSpace = 24 if copyPackages else 12
        if copyPackages:
            self.setupMessage("Hardlinks are unavailable on this storage. Packages will use copies, so a fresh setup requires at least 24 GiB free plus space for projects and checkpoints.", onLine)
        if not (self.root / ".venv").exists() and shutil.disk_usage(self.root).free < minimumSpace * 1024**3:
            raise ValidationError(f"Keep at least {minimumSpace} GiB free for this managed setup, plus space for models, caches, and checkpoints.")
        python = self.preparePython(executable, runner, cancel, onLine)
        self.readyPath.unlink(missing_ok=True)
        archive = self.downloadSource(cancel, onLine)
        temporaryTrainer = self.root / "trainer.pending"
        if temporaryTrainer.exists():
            shutil.rmtree(temporaryTrainer)
        extractTrainer(archive, temporaryTrainer)
        if cancel.is_set():
            raise InterruptedError("Setup canceled.")
        if self.trainerPath.exists():
            shutil.rmtree(self.trainerPath)
        temporaryTrainer.rename(self.trainerPath)
        for original, destination in RUNTIME_FILES.items():
            shutil.copyfile(ASSET_FOLDER / original, self.root / destination)
        self.syncDependencies(executable, python, copyPackages, runner, cancel, onLine)
        result = self.probe(runner, cancel, onLine, requireInstalled=False)
        runtimeContract.writeData(self.readyPath, {"schemaVersion": 1, "runtimeId": RUNTIME_ID, "fingerprint": self.fingerprint, "trainerRevision": TRAINER_REVISION, "python": RUNTIME_PYTHON, "pythonExecutable": str(python), "packageLinkMode": "copy" if copyPackages else "default"})
        return result

    def verifyInstallation(self, cancel: threading.Event) -> None:
        for original, installed in RUNTIME_FILES.items():
            if runtimeContract.isLink(self.root / installed) or (self.root / installed).read_bytes().replace(b"\r\n", b"\n") != (ASSET_FOLDER / original).read_bytes().replace(b"\r\n", b"\n"):
                raise ValidationError("The managed runtime files changed. Run Repair setup before training.")
        archive = self.root / "trainer.zip"
        if runtimeContract.fileDigest(archive, cancel) != TRAINER_ARCHIVE_HASH:
            raise ValidationError("The cached trainer archive changed. Run Repair setup before training.")
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                if cancel.is_set():
                    raise InterruptedError("Runtime verification canceled.")
                if entry.is_dir():
                    continue
                path = self.trainerPath.joinpath(*pathlib.PurePosixPath(entry.filename).parts[1:])
                if runtimeContract.isLink(path) or not path.is_file() or hashlib.sha256(source.read(entry)).hexdigest() != runtimeContract.fileDigest(path, cancel):
                    raise ValidationError(f"Pinned trainer source changed: {path.name}. Run Repair setup before training.")

    def probe(self, runner: ProcessRunner, cancel: threading.Event, onLine, requireInstalled: bool = True, logPath: pathlib.Path | None = None) -> dict:
        if requireInstalled and not self.installed():
            raise ValidationError("Set up the managed training runtime first.")
        self.prepareProcessFolders()
        self.probePath.unlink(missing_ok=True)
        onLine("Verifying the pinned trainer source and runtime files…")
        self.verifyInstallation(cancel)
        onLine("Checking trainer imports, bundled tokenizers, and CUDA support…")
        code = runner.run(self.command(self.root / "runtimeProbe.py", str(self.trainerPath), str(self.probePath)), self.trainerPath, self.environment(), logPath or self.logPath, cancel, onLine, timeoutSeconds=180)
        if cancel.is_set():
            raise InterruptedError("Runtime check canceled.")
        if code != 0 or not self.probePath.exists():
            raise ValidationError(f"The isolated trainer could not be imported (exit {code}). Run Repair setup. See {logPath or self.logPath}.")
        result = runtimeContract.readData(self.probePath)
        if not result.get("importsReady"):
            raise ValidationError("The trainer dependency check did not complete.")
        onLine(result.get("message", "Runtime check completed."))
        return result
