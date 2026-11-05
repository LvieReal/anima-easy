# Third-party notices

Anima Easy is inspired by [Citron’s Anima LoRA Trainer](https://github.com/citronlegacy/citron-anima-lora-trainer-ui), including its training workflow and initial hyperparameter defaults. Citron’s notice is retained below. The application implementation is new; Gradio and Citron’s setup scripts are not bundled.

The managed training backend is [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts), licensed under Apache-2.0, pinned to revision `cb70b96858477e716a2096292ab8aab433277f38`. Its source is downloaded only after explicit setup confirmation. The verified source archive is extracted with its original license and notices intact; it is not included in this application source distribution. Anima Easy adds an integration bridge rather than rewriting the downloaded source. Model weights are not bundled and remain under their respective licenses.

Runtime packages retain their own licenses: Dear PyGui (MIT), filelock (MIT), NumPy (BSD-3-Clause), Pillow (MIT-CMU), platformdirs (MIT), psutil (BSD-3-Clause), and safetensors (Apache-2.0). Dear PyGui also includes third-party components; their notices are distributed with its package. These dependencies are resolved by uv, not vendored here.

The isolated training environment has additional dependencies, including PyTorch, Accelerate, Transformers, Diffusers, bitsandbytes, and NVIDIA CUDA runtime components. They retain their respective licenses and package notices. The application's MIT license does not relicense those packages or grant rights to model weights or datasets.

## Citron

MIT License

Copyright (c) 2026 Citron Legacy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
