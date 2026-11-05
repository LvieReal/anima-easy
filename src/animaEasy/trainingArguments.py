import json
import pathlib

from animaEasy.domain import TrainingSettings
from animaEasy.errors import ValidationError
from animaEasy.trainingAssets import runtimeContract


def writeDatasetConfig(folder: pathlib.Path, settings: TrainingSettings) -> pathlib.Path:
    metadataPath = str(folder / "inputs" / "captions.json")
    configPath = folder / "inputs" / "dataset.toml"
    content = f"[general]\nshuffle_caption = false\ncaption_dropout_rate = 0.0\n\n[[datasets]]\nresolution = {settings.resolution}\nbatch_size = {settings.batchSize}\nenable_bucket = true\nmin_bucket_reso = 256\nmax_bucket_reso = 2048\nbucket_reso_steps = 64\nbucket_no_upscale = false\n\n[[datasets.subsets]]\nmetadata_file = {json.dumps(metadataPath, ensure_ascii=False)}\nnum_repeats = {settings.repeats}\n"
    configPath.write_text(content, encoding="utf-8")
    return configPath


def trainingArguments(folder: pathlib.Path, recipe: dict, resume: dict | None = None) -> list[str]:
    settings = TrainingSettings.fromData(recipe["project"]["training"])
    completedEpochs = resume["completedEpochs"] if resume else 0
    if not 0 <= completedEpochs < settings.epochs:
        raise ValidationError("This state has already reached the captured epoch target. Choose an earlier epoch state or start a new run.")
    configPath = writeDatasetConfig(folder, settings)
    arguments = ["--pretrained_model_name_or_path", recipe["models"]["diffusion"]["file"]["path"], "--qwen3", recipe["models"]["textEncoder"]["file"]["path"], "--vae", recipe["models"]["vae"]["file"]["path"], "--dataset_config", str(configPath), "--output_dir", str(folder / "outputs"), "--output_name", "anima", "--save_model_as", "safetensors", "--network_module", "networks.lora_anima", "--network_train_unet_only", "--network_dim", str(settings.rank), "--network_alpha", str(settings.alpha), "--learning_rate", str(settings.learningRate), "--max_train_epochs", str(settings.epochs - completedEpochs), "--gradient_accumulation_steps", str(settings.gradientAccumulation), "--mixed_precision", settings.mixedPrecision, "--save_precision", "float" if settings.mixedPrecision == "no" else settings.mixedPrecision, "--optimizer_type", settings.optimizer, "--lr_scheduler", "constant", "--seed", str(settings.seed), "--max_data_loader_n_workers", "0", "--attn_mode", "torch", "--split_attn", "--save_every_n_epochs", "1", "--save_last_n_epochs", str(settings.keepCheckpoints), "--save_last_n_epochs_state", str(settings.keepCheckpoints), "--save_state", "--save_state_on_train_end", "--vae_batch_size", "1", "--text_encoder_batch_size", "1", "--console_log_simple"]
    if settings.gradientCheckpointing:
        arguments.append("--gradient_checkpointing")
    if settings.cacheLatents:
        arguments.extend(["--cache_latents", "--cache_latents_to_disk"])
    if settings.cacheTextEncoder:
        arguments.extend(["--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk"])
    if settings.blocksToSwap:
        arguments.extend(["--blocks_to_swap", str(settings.blocksToSwap)])
    if settings.vaeChunkSize:
        arguments.extend(["--vae_chunk_size", str(settings.vaeChunkSize)])
    if resume:
        arguments.extend(["--resume", resume["path"], "--initial_step", "0"])
    runtimeContract.writeData(folder / "inputs" / "arguments.json", {"schemaVersion": 1, "arguments": arguments, "completedEpochsBeforeResume": completedEpochs})
    return arguments
