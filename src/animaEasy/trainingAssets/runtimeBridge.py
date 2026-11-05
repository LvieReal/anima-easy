import _thread
import math
import os
import pathlib
import sys
import threading
import time


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import runtimeContract


class BridgeContext:
    def __init__(self, specification: dict):
        self.specification = specification
        self.folder = pathlib.Path(specification["runFolder"]).resolve()
        self.recipe = runtimeContract.readData(self.folder / "inputs" / "recipe.json")
        if runtimeContract.fileDigest(self.folder / "inputs" / "recipe.json") != specification["recipeHash"]:
            raise ValueError("The run configuration changed before launch.")
        if runtimeContract.fileDigest(self.folder / "inputs" / "captions.json") != self.recipe["captionHash"]:
            raise ValueError("The captured captions changed before launch.")
        self.stateKey = pathlib.Path(specification["stateKeyPath"]).read_bytes()
        self.resume = specification.get("resume")
        self.epochOffset = self.resume["completedEpochs"] if self.resume else 0
        self.stepOffset = self.resume["completedSteps"] if self.resume else 0
        self.targetEpochs = self.recipe["project"]["training"]["epochs"]
        self.localStep = 0
        self.localTotal = 0
        self.currentEpoch = self.epochOffset + 1
        self.completedEpochs = self.epochOffset
        self.lastLoss = None
        self.lastEvent = 0
        self.startedAt = time.monotonic()
        self.finalModelSaved = False
        self.pendingFinalState = None

    def emit(self, kind: str, **values) -> None:
        runtimeContract.emitEvent(self.specification["token"], kind, **values)

    def progress(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.lastEvent < 0.25:
            return
        self.lastEvent = now
        self.emit("progress", step=self.localStep + self.stepOffset, total=self.localTotal + self.stepOffset, epoch=self.currentEpoch, epochs=self.targetEpochs, loss=self.lastLoss, elapsedSeconds=now - self.startedAt)

    def saveStateMarker(self, folder: pathlib.Path, localEpoch: int) -> None:
        self.emit("phase", message="Verifying the completed epoch state…")
        self.completedEpochs = localEpoch + self.epochOffset
        runtimeContract.sealState(folder, self.specification["runId"], self.specification["recipeHash"], self.completedEpochs, self.localStep + self.stepOffset, self.stateKey)
        self.emit("stateSaved", name=folder.name, completedEpochs=self.completedEpochs, completedSteps=self.localStep + self.stepOffset)


def createAdapter(context: BridgeContext):
    sys.path.insert(0, context.specification["trainerPath"])
    import anima_train_network
    import library.checkpoint_io
    import library.dataset
    import library.strategy_anima
    import train_network

    cacheFolder = context.folder / "cache"
    cacheFolder.mkdir(exist_ok=True)
    originalEpoch = library.dataset.BaseDataset.set_current_epoch
    originalStep = library.dataset.BaseDataset.set_current_step
    originalSeed = library.dataset.BaseDataset.set_seed
    originalSaveEpoch = library.checkpoint_io.save_and_remove_state_on_epoch_end
    originalSaveFinal = library.checkpoint_io.save_state_on_train_end
    originalEpochName = library.checkpoint_io.get_epoch_ckpt_name

    class ManagedLatents(library.strategy_anima.AnimaLatentsCachingStrategy):
        def getLatentsPath(self, imagePath, imageSize):
            return str(runtimeContract.cacheStem(cacheFolder, imagePath)) + f"_{imageSize[0]:04d}x{imageSize[1]:04d}_anima.npz"

    ManagedLatents.get_latents_npz_path = ManagedLatents.getLatentsPath

    class ManagedTextOutputs(library.strategy_anima.AnimaTextEncoderOutputsCachingStrategy):
        def getOutputsPath(self, imagePath):
            return str(runtimeContract.cacheStem(cacheFolder, imagePath)) + "_anima_te.npz"

    ManagedTextOutputs.get_outputs_npz_path = ManagedTextOutputs.getOutputsPath

    class ObservedProgress(train_network.tqdm):
        def __init__(self, *arguments, **values):
            self.observed = values.get("desc") == "steps"
            super().__init__(*arguments, **values)
            if self.observed:
                context.localTotal = int(self.total or 0)
                context.startedAt = time.monotonic()
                context.progress(force=True)

        def display(self, *arguments, **values):
            if not self.observed:
                return super().display(*arguments, **values)

        def update(self, amount=1):
            result = super().update(amount)
            if self.observed:
                context.localStep = int(self.n)
                context.progress(force=self.n == self.total)
            return result

        def setPostfix(self, *arguments, **values):
            if self.observed and "avr_loss" in values:
                loss = float(values["avr_loss"])
                if not math.isfinite(loss):
                    raise FloatingPointError("Training loss became non-finite. Try a lower learning rate or a different mixed-precision setting in a new run.")
                context.lastLoss = loss
                context.progress()
            return super().set_postfix(*arguments, **values)

    ObservedProgress.set_postfix = ObservedProgress.setPostfix
    train_network.tqdm = ObservedProgress

    def setTrainingEpoch(dataset, epoch):
        context.currentEpoch = epoch + context.epochOffset
        return originalEpoch(dataset, epoch + context.epochOffset)

    def setDatasetSeed(dataset, seed):
        originalSeed(dataset, seed)
        if context.epochOffset:
            originalEpoch(dataset, context.epochOffset)

    def setTrainingStep(dataset, step):
        return originalStep(dataset, step + context.stepOffset)

    def saveEpoch(arguments, accelerator, epoch):
        context.emit("phase", message="Saving the epoch checkpoint and trainer state…")
        result = originalSaveEpoch(arguments, accelerator, epoch + context.epochOffset)
        name = library.checkpoint_io.EPOCH_STATE_NAME.format(arguments.output_name, epoch + context.epochOffset)
        context.saveStateMarker(pathlib.Path(arguments.output_dir) / name, epoch)
        return result

    def saveFinal(arguments, accelerator):
        context.emit("phase", message="Saving the final trainer state…")
        result = originalSaveFinal(arguments, accelerator)
        name = library.checkpoint_io.LAST_STATE_NAME.format(arguments.output_name)
        context.pendingFinalState = pathlib.Path(arguments.output_dir) / name
        return result

    def epochCheckpointName(arguments, extension, epoch):
        return originalEpochName(arguments, extension, epoch + context.epochOffset)

    library.checkpoint_io.get_epoch_ckpt_name = epochCheckpointName
    library.dataset.BaseDataset.set_current_epoch = setTrainingEpoch
    library.dataset.BaseDataset.set_current_step = setTrainingStep
    library.dataset.BaseDataset.set_seed = setDatasetSeed
    library.checkpoint_io.save_and_remove_state_on_epoch_end = saveEpoch
    library.checkpoint_io.save_state_on_train_end = saveFinal

    class ManagedTrainer(anima_train_network.AnimaNetworkTrainer):
        def getLatentsStrategy(self, arguments):
            return ManagedLatents(arguments.cache_latents_to_disk, arguments.vae_batch_size, arguments.skip_cache_check)

        def getTextStrategy(self, arguments):
            if arguments.cache_text_encoder_outputs:
                return ManagedTextOutputs(arguments.cache_text_encoder_outputs_to_disk, arguments.text_encoder_batch_size, arguments.skip_cache_check, False)
            return None

        def saveModel(self, **values):
            context.emit("phase", message="Saving a LoRA checkpoint…")
            adjusted = {**values, "steps": values["steps"] + context.stepOffset, "epoch_no": values["epoch_no"] + context.epochOffset}
            self._metadata["animaEasyTrainerRevision"] = context.recipe["trainerRevision"]
            self._metadata["animaEasyRunId"] = context.specification["runId"]
            self._metadata["ss_sd_scripts_commit_hash"] = context.recipe["trainerRevision"]
            self._metadata["ss_num_epochs"] = str(context.targetEpochs)
            self._metadata["ss_max_train_epochs"] = str(context.targetEpochs)
            self._metadata["ss_max_train_steps"] = str(context.localTotal + context.stepOffset)
            result = super()._save_model(**adjusted)
            path = pathlib.Path(values["args"].output_dir) / values["ckpt_name"]
            runtimeContract.writeData(path.with_name(path.name + ".saved.json"), {"schemaVersion": 1, "runId": context.specification["runId"], "recipeHash": context.specification["recipeHash"], "sizeBytes": path.stat().st_size, "sha256": runtimeContract.fileDigest(path), "completedSteps": adjusted["steps"], "completedEpochs": adjusted["epoch_no"]})
            context.emit("checkpointSaved", name=path.name)
            if values.get("force_sync_upload"):
                if context.pendingFinalState is None or adjusted["epoch_no"] != context.targetEpochs or context.localStep != context.localTotal:
                    raise RuntimeError("The final checkpoint does not match a fully saved training target.")
                context.saveStateMarker(context.pendingFinalState, values["epoch_no"])
                context.finalModelSaved = True
            return result

        def removeModel(self, **values):
            result = super()._remove_model(**values)
            path = pathlib.Path(values["args"].output_dir) / values["old_ckpt_name"]
            if not path.exists():
                path.with_name(path.name + ".saved.json").unlink(missing_ok=True)
            return result

    ManagedTrainer.get_latents_caching_strategy = ManagedTrainer.getLatentsStrategy
    ManagedTrainer.get_text_encoder_outputs_caching_strategy = ManagedTrainer.getTextStrategy
    ManagedTrainer._save_model = ManagedTrainer.saveModel
    ManagedTrainer._remove_model = ManagedTrainer.removeModel
    return ManagedTrainer(), anima_train_network.setup_parser().parse_args(context.specification["arguments"])


def watchOwner(specification: dict, finished: threading.Event) -> None:
    import psutil

    stopPath = pathlib.Path(specification["runFolder"]) / "stop.request"
    while not finished.wait(0.25):
        try:
            owner = psutil.Process(specification["ownerPid"])
            ownerAlive = owner.is_running() and abs(owner.create_time() - specification["ownerCreatedAt"]) < 0.01 and owner.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            ownerAlive = False
        if stopPath.exists() or not ownerAlive:
            _thread.interrupt_main()
            if not ownerAlive and not finished.wait(10):
                os._exit(130)
            return


def main() -> None:
    specification = runtimeContract.readData(pathlib.Path(sys.argv[1]))
    finished = threading.Event()
    watcher = threading.Thread(target=watchOwner, args=(specification, finished), name="Training owner guard", daemon=True)
    watcher.start()
    try:
        context = BridgeContext(specification)
        context.emit("phase", message="Loading the pinned Anima trainer…")
        trainer, arguments = createAdapter(context)
        import library.args
        import torch

        library.args.verify_command_line_training_args(arguments)
        if not torch.cuda.is_available():
            raise RuntimeError("A usable NVIDIA CUDA GPU is required. Training did not start.")
        if arguments.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported on this GPU. Choose FP16 in a new run.")
        if context.resume:
            state = context.resume
            context.emit("phase", message="Verifying the saved optimizer and trainer state…")
            verified = runtimeContract.verifyState(pathlib.Path(state["path"]), state["runId"], specification["recipeHash"], context.stateKey)
            if verified["completedEpochs"] != context.epochOffset or verified["completedSteps"] != context.stepOffset:
                raise ValueError("The selected resume state changed.")
        context.emit("phase", message="Loading models and preparing run-owned caches. No step progress is available yet.")
        trainer.train(arguments)
        if not context.finalModelSaved or context.localTotal < 1 or context.localStep != context.localTotal or context.completedEpochs != context.targetEpochs:
            raise RuntimeError("The trainer returned without a complete final checkpoint and the requested number of updates.")
        context.progress(force=True)
        context.emit("completed", completedSteps=context.localStep + context.stepOffset, completedEpochs=context.completedEpochs)
    except KeyboardInterrupt:
        runtimeContract.emitEvent(specification["token"], "stopped", message="Training stopped. Only completed, verified epoch states can be resumed.")
        raise SystemExit(130) from None
    finally:
        finished.set()


if __name__ == "__main__":
    main()
