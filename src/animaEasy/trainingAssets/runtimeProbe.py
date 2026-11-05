import importlib.metadata
import json
import pathlib
import sys
import tomllib


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import runtimeContract


def inspectRuntime(trainerPath: pathlib.Path) -> dict:
    sys.path.insert(0, str(trainerPath))
    import packaging.requirements
    import packaging.utils

    root = pathlib.Path(__file__).resolve().parent
    projectPath = root / "pyproject.toml" if (root / "pyproject.toml").exists() else root / "runtimeProject.toml"
    lockPath = root / "uv.lock" if (root / "uv.lock").exists() else root / "runtime.lock"
    profile = tomllib.loads(projectPath.read_text(encoding="utf-8"))["project"]
    if profile["requires-python"] != f"=={sys.version.split()[0]}":
        raise RuntimeError("The managed Python version changed. Run Repair setup before training.")
    locked = {packaging.utils.canonicalize_name(item["name"]): item["version"] for item in tomllib.loads(lockPath.read_text(encoding="utf-8"))["package"]}
    for distribution in importlib.metadata.distributions():
        name = packaging.utils.canonicalize_name(distribution.metadata["Name"])
        if locked.get(name) != distribution.version:
            raise RuntimeError(f"The managed package {name} changed. Run Repair setup before training.")
    for value in profile["dependencies"]:
        requirement = packaging.requirements.Requirement(value)
        if importlib.metadata.version(requirement.name) not in requirement.specifier:
            raise RuntimeError(f"The managed dependency {requirement.name} does not match its pinned version.")
    import accelerate
    import anima_train_network
    import diffusers
    import psutil
    import torch
    import transformers
    from library import anima_utils

    anima_train_network.setup_parser()
    anima_utils.load_qwen3_tokenizer("registeredModel.safetensors")
    anima_utils.load_t5_tokenizer()
    data = {"schemaVersion": 1, "importsReady": True, "python": sys.version.split()[0], "torch": torch.__version__, "cudaVersion": torch.version.cuda, "accelerate": accelerate.__version__, "transformers": transformers.__version__, "diffusers": diffusers.__version__, "systemMemoryBytes": psutil.virtual_memory().total, "cudaReady": False, "bf16Supported": False, "message": "Runtime installed. No usable NVIDIA CUDA GPU was detected; training is blocked until the GPU check passes."}
    if torch.cuda.is_available():
        try:
            device = torch.cuda.get_device_properties(0)
            sample = torch.ones((32, 32), device="cuda")
            result = sample @ sample
            torch.cuda.synchronize()
            if not torch.isfinite(result).all().item():
                raise RuntimeError("The CUDA kernel check returned non-finite values.")
            del sample, result
            torch.cuda.empty_cache()
            freeBytes, totalBytes = torch.cuda.mem_get_info(0)
            data.update({"cudaReady": True, "deviceName": device.name, "capability": list(torch.cuda.get_device_capability(0)), "architectures": torch.cuda.get_arch_list(), "freeMemoryBytes": freeBytes, "totalMemoryBytes": totalBytes, "bf16Supported": torch.cuda.is_bf16_supported(), "message": f"CUDA check passed: {device.name}. This is a kernel check, not a guarantee that the selected training settings fit in VRAM."})
        except Exception as error:
            data["message"] = f"CUDA check failed: {error}. Update or repair the NVIDIA driver before training."
    return data


if __name__ == "__main__":
    result = inspectRuntime(pathlib.Path(sys.argv[1]).resolve())
    runtimeContract.writeData(pathlib.Path(sys.argv[2]), result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
