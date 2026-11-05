import math
import multiprocessing
import pathlib
import random
import sys


sys.path.insert(0, sys.argv[2])
import runtimeBridge
import runtimeContract


def main() -> None:
    specificationPath = pathlib.Path(sys.argv[1])
    specification = runtimeContract.readData(specificationPath)
    context = runtimeBridge.BridgeContext(specification)
    trainer, arguments = runtimeBridge.createAdapter(context)
    import accelerate
    import library.accelerator_setup
    import library.anima_models
    import library.args
    import library.checkpoint_io
    import library.config_util
    import library.dataset
    import library.optimizer
    import library.strategy_base
    import library.utils
    import networks.lora_anima
    import numpy
    import safetensors
    import torch
    import train_network

    library.utils.setup_logging(arguments, reset=True)
    library.args.verify_command_line_training_args(arguments)
    library.args.verify_training_args(arguments)
    library.accelerator_setup.prepare_dataset_args(arguments, True)
    accelerate.utils.set_seed(arguments.seed)
    torch.set_num_threads(1)
    tokenization = trainer.get_tokenize_strategy(arguments)
    library.strategy_base.TokenizeStrategy.set_strategy(tokenization)
    latentStrategy = trainer.get_latents_caching_strategy(arguments)
    library.strategy_base.LatentsCachingStrategy.set_strategy(latentStrategy)
    textStrategy = trainer.get_text_encoder_outputs_caching_strategy(arguments)
    generator = library.config_util.BlueprintGenerator(library.config_util.ConfigSanitizer(True, True, arguments.masked_loss, True))
    configuration = library.config_util.load_user_config(arguments.dataset_config)
    blueprint = generator.generate(configuration, arguments)
    datasetGroup, validationGroup = library.config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    trainer.assert_extra_args(arguments, datasetGroup, validationGroup)
    metadata = runtimeContract.readData(context.folder / "inputs" / "captions.json")
    cachePaths = set()
    textPaths = set()
    for dataset in datasetGroup.datasets:
        assert len(dataset.image_data) == len(metadata)
        assert dataset.current_epoch == context.epochOffset
        dataset.debug_dataset = True
        for info in dataset.image_data.values():
            assert info.caption == metadata[info.image_key]["caption"]
            assert list(info.image_size) == metadata[info.image_key]["image_size"]
            assert info.latents_npz is None
            assert not info.absolute_path.endswith(".npz")
            latentPath = pathlib.Path(latentStrategy.get_latents_npz_path(info.absolute_path, info.image_size))
            textPath = pathlib.Path(textStrategy.get_outputs_npz_path(info.absolute_path))
            assert latentPath.parent == context.folder / "cache"
            assert textPath.parent == context.folder / "cache"
            cachePaths.add(latentPath)
            textPaths.add(textPath)
            encoded = tokenization.tokenize(info.caption)
            assert len(encoded) == 4
    assert len(cachePaths) == len(metadata)
    assert len(textPaths) == len(metadata)
    cachePath = next(iter(cachePaths))
    values = torch.arange(16 * 32 * 32, dtype=torch.float32).reshape(16, 32, 32)
    latentStrategy.save_latents_to_disk(str(cachePath), values, [64, 96], [0, 0, 64, 96], key_reso_suffix="_32x32")
    assert latentStrategy.is_disk_cached_latents_expected((256, 256), str(cachePath), False, False)
    loaded = latentStrategy.load_latents_from_disk(str(cachePath), (256, 256))
    numpy.testing.assert_array_equal(loaded[0], values.numpy())
    assert loaded[1] == [64, 96]
    assert arguments.network_module == "networks.lora_anima"
    assert arguments.network_train_unet_only
    assert arguments.max_data_loader_n_workers == 0
    assert arguments.save_state and arguments.save_state_on_train_end
    assert arguments.lr_scheduler == "constant"
    assert not arguments.fp8_base and not arguments.fp8_scaled
    torch.manual_seed(731)
    random.seed(731)
    numpy.random.seed(731)
    model = library.anima_models.Anima(32, 32, 1, 16, 16, 2, 1, concat_padding_mask=False, model_channels=64, num_blocks=2, num_heads=4, crossattn_emb_channels=32, pos_emb_cls="rope3d", use_llm_adapter=False)
    model.requires_grad_(False)
    network = networks.lora_anima.create_network(1.0, arguments.network_dim, arguments.network_alpha, None, [], model)
    network.apply_to([], model, False, True)
    network.requires_grad_(True)
    assert len(network.unet_loras) > 0
    accelerator = accelerate.Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=arguments.gradient_accumulation_steps)
    stepsPerEpoch = math.ceil(len(datasetGroup) / arguments.gradient_accumulation_steps)
    arguments.max_train_steps = stepsPerEpoch * arguments.max_train_epochs
    optimizer = library.optimizer.get_optimizer(arguments, network.parameters())[2]
    scheduler = library.optimizer.get_scheduler_fix(arguments, optimizer, 1)
    currentEpoch, currentStep = multiprocessing.Value("i", 1), multiprocessing.Value("i", 0)
    datasetGroup.set_current_strategies()
    collator = library.dataset.collator_class(currentEpoch, currentStep, datasetGroup)
    dataLoader = torch.utils.data.DataLoader(datasetGroup, batch_size=1, shuffle=True, collate_fn=collator, num_workers=0)
    network, optimizer, dataLoader, scheduler = accelerator.prepare(network, optimizer, dataLoader, scheduler)
    projection = accelerator.unwrap_model(network).unet_loras[0]
    initialWeights = projection.lora_up.weight.detach().clone()
    progress = train_network.tqdm(range(arguments.max_train_steps), desc="steps")

    def trainEpoch(epoch: int) -> list:
        currentEpoch.value = epoch
        batches = []
        for batch in dataLoader:
            currentStep.value = progress.n
            batches.append({"keys": batch["image_keys"], "captions": batch["captions"]})
            with accelerator.accumulate(network):
                sample = torch.randn(3, projection.lora_down.in_features) + batch["images"].mean()
                loss = projection(sample).square().mean()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress.update(1)
            progress.set_postfix(avr_loss=float(loss.detach()))
        assert all(dataset.current_epoch == epoch + context.epochOffset for dataset in datasetGroup.datasets)
        return batches

    if context.resume:
        saved = runtimeContract.verifyState(pathlib.Path(context.resume["path"]), context.resume["runId"], specification["recipeHash"], context.stateKey)
        assert saved["completedEpochs"] == context.epochOffset
        assert arguments.max_train_epochs == context.targetEpochs - context.epochOffset
        assert arguments.initial_step == 0
        accelerator.load_state(context.resume["path"])
        expected = runtimeContract.readData(pathlib.Path(specification["expectedPath"]))
        assert random.random() == expected["nextPythonRandom"]
        assert float(numpy.random.random()) == expected["nextNumpyRandom"]
        assert trainEpoch(1) == expected["nextBatches"]
        numpy.testing.assert_array_equal(projection.lora_up.weight.detach().numpy(), numpy.asarray(expected["nextWeights"], dtype=numpy.float32))
        assert int(next(iter(optimizer.state.values()))["step"]) == expected["nextOptimizerStep"]
        assert scheduler.state_dict() == expected["nextSchedulerState"]
    else:
        trainEpoch(1)
    assert not torch.equal(initialWeights, projection.lora_up.weight)
    expectedEpoch = context.epochOffset + 1
    checkpointName = library.checkpoint_io.get_epoch_ckpt_name(arguments, ".safetensors", 1)
    trainer._metadata = {"ss_network_module": arguments.network_module, "ss_network_dim": str(arguments.network_dim), "ss_network_alpha": str(arguments.network_alpha)}
    trainer._minimum_metadata = dict(trainer._metadata)
    trainer._save_model(args=arguments, accelerator=accelerator, save_dtype=torch.float32, ckpt_name=checkpointName, unwrapped_nw=accelerator.unwrap_model(network), steps=stepsPerEpoch, epoch_no=1)
    with safetensors.safe_open(pathlib.Path(arguments.output_dir) / checkpointName, framework="pt", device="cpu") as weights:
        assert len(weights.keys()) > 0
        assert weights.metadata()["ss_steps"] == str(stepsPerEpoch + context.stepOffset)
        assert weights.metadata()["ss_epoch"] == str(expectedEpoch)
        assert weights.metadata()["ss_max_train_epochs"] == str(context.targetEpochs)
        assert weights.metadata()["ss_num_epochs"] == str(context.targetEpochs)
        assert weights.metadata()["ss_sd_scripts_commit_hash"] == context.recipe["trainerRevision"]
        assert weights.metadata()["animaEasyTrainerRevision"] == context.recipe["trainerRevision"]
    library.checkpoint_io.save_and_remove_state_on_epoch_end(arguments, accelerator, 1)
    stateName = library.checkpoint_io.EPOCH_STATE_NAME.format(arguments.output_name, expectedEpoch)
    statePath = pathlib.Path(arguments.output_dir) / stateName
    sealed = runtimeContract.verifyState(statePath, specification["runId"], specification["recipeHash"], context.stateKey)
    assert sealed["completedSteps"] == stepsPerEpoch + context.stepOffset
    assert sealed["completedEpochs"] == expectedEpoch
    finalStateName = None
    if not context.resume:
        expected = {"nextPythonRandom": random.random(), "nextNumpyRandom": float(numpy.random.random())}
        expected["nextBatches"] = trainEpoch(2)
        expected.update({"nextWeights": projection.lora_up.weight.detach().tolist(), "nextOptimizerStep": int(next(iter(optimizer.state.values()))["step"]), "nextSchedulerState": scheduler.state_dict()})
        runtimeContract.writeData(context.folder / "cpuExpected.json", expected)
    else:
        for epoch in range(2, arguments.max_train_epochs + 1):
            trainEpoch(epoch)
        library.checkpoint_io.save_state_on_train_end(arguments, accelerator)
        finalStateName = library.checkpoint_io.LAST_STATE_NAME.format(arguments.output_name)
        finalStatePath = pathlib.Path(arguments.output_dir) / finalStateName
        assert (finalStatePath / "optimizer.bin").is_file()
        assert not (finalStatePath / runtimeContract.STATE_MARKER).exists()
        assert not context.finalModelSaved
        finalName = library.checkpoint_io.get_last_ckpt_name(arguments, ".safetensors")
        finalPath = pathlib.Path(arguments.output_dir) / finalName
        finalPath.mkdir()
        finalArguments = {"args": arguments, "accelerator": accelerator, "save_dtype": torch.float32, "ckpt_name": finalName, "unwrapped_nw": accelerator.unwrap_model(network), "steps": context.localStep, "epoch_no": arguments.max_train_epochs, "force_sync_upload": True}
        try:
            trainer._save_model(**finalArguments)
        except (OSError, safetensors.SafetensorError):
            pass
        else:
            raise AssertionError("Saving weights over a directory must fail.")
        assert not (finalStatePath / runtimeContract.STATE_MARKER).exists()
        assert not context.finalModelSaved
        finalPath.rmdir()
        trainer._save_model(**finalArguments)
        assert context.finalModelSaved
        finalState = runtimeContract.verifyState(finalStatePath, specification["runId"], specification["recipeHash"], context.stateKey)
        assert finalState["completedEpochs"] == context.targetEpochs
        assert finalState["completedSteps"] == stepsPerEpoch * context.targetEpochs
        trainer._remove_model(args=arguments, accelerator=accelerator, old_ckpt_name=checkpointName)
        assert not (pathlib.Path(arguments.output_dir) / checkpointName).exists()
        assert not (pathlib.Path(arguments.output_dir) / (checkpointName + ".saved.json")).exists()
    progress.close()
    runtimeContract.writeData(context.folder / "cpuContractResult.json", {"schemaVersion": 1, "stateName": stateName, "stepsPerEpoch": stepsPerEpoch, "checkpointName": checkpointName, "loraModules": len(accelerator.unwrap_model(network).unet_loras), "completedEpochs": expectedEpoch, "finalStateName": finalStateName})
    print("Real trainer parser, dataset iteration, tokenizers, cache IO, Anima LoRA projection, optimizer, and saved-state contracts passed on CPU. No full Anima training run or GPU fit was tested.", flush=True)


if __name__ == "__main__":
    main()
