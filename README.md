# flux-fast
Making Flux go brrr on GPUs. With simple recipes from this repo, we enabled ~2.5x speedup on Flux.1-Schnell and Flux.1-Dev using (mainly) pure PyTorch code and a beefy GPU like H100. This repo is NOT meant to be a library or an out-of-the-box solution. So, please fork the repo, hack into the code, and share your results 🤗

Check out the accompanying blog post [here](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/).

**Updates**

**July 1, 2025**: This repository now supports AMD MI300X GPUs using AITER kernels [(PR)](https://github.com/huggingface/flux-fast/pull/10). The README has been updated to provide instructions on how to run on AMD GPUs.

**June 28, 2025**: This repository now supports [Flux.1 Kontext Dev](https://hf.co/black-forest-labs/FLUX.1-Kontext-dev). We enabled ~2.5x speedup on it. Check out [this section](#flux1-kontext-dev) for more details.

## Results

<table>
  <thead>
    <tr>
      <th>Description</th>
      <th>Image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Flux.1-Schnell</td>
      <td><img src="https://github.com/user-attachments/assets/3f18d621-bdcd-423d-a66c-fd34bbd90f27" width=500 alt="new_flux_schnell_plot" /></td>
    </tr>
    <tr>
      <td>Flux.1-Dev</td>
      <td><img src="https://github.com/user-attachments/assets/48945137-c826-497a-a292-b1f976a5b16a" width=500 alt="flux_dev_result_plot" /></td>
    </tr>
  </tbody>
</table>


Summary of the optimizations:
* Running with the bfloat16 precision
* `torch.compile`
* Combining q,k,v projections for attention computation
* `torch.channels_last` memory format for the decoder output
* Flash Attention v3 (FA3) with (unscaled) conversion of inputs to `torch.float8_e4m3fn`
* Dynamic float8 quantization and quantization of Linear layer weights via `torchao`'s `float8_dynamic_activation_float8_weight`
* Inductor flags:
    * `conv_1x1_as_mm = True`
    * `epilogue_fusion = False`
    * `coordinate_descent_tuning = True`
    * `coordinate_descent_check_all_directions = True`
* `torch.export` + Ahead-of-time Inductor (AOTI) + CUDAGraphs
* Almost lossless cache acceleration with `cache-dit: DBCache + F12B12`

All of the above optimizations are lossless (outside of minor numerical differences sometimes
introduced through the use of `torch.compile` / `torch.export`) EXCEPT FOR dynamic float8 quantization.
Disable quantization if you want the same quality results as the baseline while still being
quite a bit faster.

Here are some example outputs with Flux.1-Schnell for prompt `"A cat playing with a ball of yarn"`:

<table>
  <thead>
    <tr>
      <th>Configuration</th>
      <th>Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Baseline</strong></td>
      <td><img src="https://github.com/user-attachments/assets/8ba746d2-fbf3-4e30-adc4-11303231c146" alt="baseline_output" width=400/></td>
    </tr>
    <tr>
      <td><strong>Fully-optimized (with quantization)</strong></td>
      <td><img src="https://github.com/user-attachments/assets/1a31dec4-38d5-45b2-8ae6-c7fb2e6413a4" alt="fast_output" width=400/></td>
    </tr>
  </tbody>
</table>

## Setup
We rely primarily on pure PyTorch for the optimizations. Currently, a relatively recent nightly version of PyTorch is required.

The numbers reported here were gathered using:

For NVIDIA:
* `torch==2.8.0.dev20250605+cu126` - note that we rely on some fixes since 2.7
* `torchao==0.12.0.dev20250610+cu126` - note that we rely on a fix in the 06/10 nightly
* `diffusers` - with [this fix](https://github.com/huggingface/diffusers/pull/11696) included
* `flash_attn_3==3.0.0b1`

For AMD:
* `torch==2.8.0.dev20250605+rocm6.4` - note that we rely on some fixes since 2.7
* `torchao==0.12.0.dev20250610+rocm6.4` - note that we rely on a fix in the 06/10 nightly
* `diffusers` - with [this fix](https://github.com/huggingface/diffusers/pull/11696) included
* `aiter-0.1.4.dev17+gd0384d4`

To install deps on NVIDIA:
```
pip install -U huggingface_hub[hf_xet] accelerate transformers
pip install -U diffusers
pip install --pre torch==2.8.0.dev20250605+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126
pip install --pre torchao==0.12.0.dev20250609+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126
```

(For NVIDIA) To install flash attention v3, follow the instructions in https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release.

To install deps on AMD:
```
pip install -U diffusers
pip install --pre torch==2.8.0.dev20250605+rocm6.4 --index-url https://download.pytorch.org/whl/nightly/rocm6.4
pip install --pre torchao==0.12.0.dev20250609+rocm6.4 --index-url https://download.pytorch.org/whl/nightly/rocm6.4
pip install git+https://github.com/ROCm/aiter
```

(For AMD) Instead of flash attention v3, we use (AITER)[https://github.com/ROCm/aiter]. It provides the required fp8 MHA kernels  

For hardware, we used a 96GB 700W H100 GPU and 192GB MI300X GPU. Some of the optimizations applied (BFloat16, torch.compile, Combining q,k,v projections, dynamic float8 quantization) are available on CPU as well.

## Run the optimized pipeline

On NVIDIA:
```sh
python gen_image.py --prompt "An astronaut standing next to a giant lemon" --output-file output.png --use-cached-model
```
This will include all optimizations and will attempt to use pre-cached binary models
generated via `torch.export` + AOTI. To generate these binaries for subsequent runs, run
the above command without the `--use-cached-model` flag.

> [!IMPORTANT]
> The binaries won't work for hardware that is sufficiently different from the hardware they were
> obtained on. For example, if the binaries were obtained on an H100, they won't work on A100.
> Further, the binaries are currently Linux-only and include dependencies on specific versions
> of system libs such as libstdc++; they will not work if they were generated in a sufficiently
> different environment than the one present at runtime. The PyTorch Compiler team is working on
> solutions for more portable binaries / artifact caching.

On AMD:
```sh
python gen_image.py --prompt "A cat playing with a ball of yarn" --output-file output.png --compile_export_mode compile
```
Currently, only torch.export is not working as expected. Instead, use `torch.compile` as shown in the above command.


## Benchmarking
[`run_benchmark.py`](./run_benchmark.py) is the main script for benchmarking the different optimization techniques.
Usage:
```
usage: run_benchmark.py [-h] [--ckpt CKPT] [--prompt PROMPT] [--cache-dir CACHE_DIR]
                        [--device {cuda,cpu}] [--num_inference_steps NUM_INFERENCE_STEPS]
                        [--output-file OUTPUT_FILE] [--trace-file TRACE_FILE] [--disable_bf16]
                        [--compile_export_mode {compile,export_aoti,disabled}]
                        [--disable_fused_projections] [--disable_channels_last] [--disable_fa3]
                        [--disable_quant] [--disable_inductor_tuning_flags]

options:
  -h, --help            show this help message and exit
  --ckpt CKPT           Model checkpoint path (default: black-forest-labs/FLUX.1-schnell)
  --prompt PROMPT       Text prompt (default: A cat playing with a ball of yarn)
  --cache-dir CACHE_DIR
                        Cache directory for storing exported models (default:
                        ~/.cache/flux-fast)
  --device {cuda,cpu}   Device to use (default: cuda)
  --num_inference_steps NUM_INFERENCE_STEPS
                        Number of denoising steps (default: 4)
  --output-file OUTPUT_FILE
                        Output image file path (default: output.png)
  --trace-file TRACE_FILE
                        Output PyTorch Profiler trace file path (default: None)
  --disable_bf16        Disables usage of torch.bfloat16 (default: False)
  --compile_export_mode {compile,export_aoti,disabled}
                        Configures how torch.compile or torch.export + AOTI are used (default:
                        export_aoti)
  --disable_fused_projections
                        Disables fused q,k,v projections (default: False)
  --disable_channels_last
                        Disables usage of torch.channels_last memory format (default: False)
  --disable_fa3         Disables use of Flash Attention V3 (default: False)
  --disable_quant       Disables usage of dynamic float8 quantization (default: False)
  --disable_inductor_tuning_flags
                        Disables use of inductor tuning flags (default: False)
```

Note that all optimizations are on by default and each can be individually toggled. Example run:
```
# Run with all optimizations and output a trace file alongside benchmark numbers
python run_benchmark.py --trace-file profiler_trace.json.gz
```

After an experiment has been run, you should expect to see
mean / variance times in seconds for 10 benchmarking runs printed to STDOUT, as well as:

* A `.png` image file corresponding to the experiment (e.g. `output.png`). The path can be configured via `--output-file`.
* An optional PyTorch profiler trace (e.g. `profiler_trace.json.gz`). The path can be configured via `--trace-file`

> [!IMPORTANT]
> For benchmarking purposes, we use reasonable defaults. For example, for all the benchmarking experiments, we use
> the 1024x1024 resolution. For Schnell, we use 4 denoising steps, and for Dev and Kontext, we use 28.

## Flux.1 Kontext Dev
We ran the exact same setup as above on [Flux.1 Kontext Dev](https://hf.co/black-forest-labs/FLUX.1-Kontext-dev) and obtained the following result:

<div align="center">
<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux_kontext_optims.png" width=500 alt="flux_kontext_plot"/>
</div><br>

Here are some example outputs for prompt `"Make Pikachu hold a sign that says 'Black Forest Labs is awesome', yarn art style, detailed, vibrant colors"` and [this image](https://huggingface.co/datasets/huggingface/documentation-images/blob/main/diffusers/yarn-art-pikachu.png):

<table>
  <thead>
    <tr>
      <th>Configuration</th>
      <th>Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Baseline</strong></td>
      <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/bf16_kontext.png" alt="baseline_output" width=400/></td>
    </tr>
    <tr>
      <td><strong>Fully-optimized (with quantization)</strong></td>
      <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/fully_optimized_kontext.png" alt="fast_output" width=400/></td>
    </tr>
  </tbody>
</table>

<details>
<summary><b>Notes</b></summary>

* You need to install `diffusers` with [this fix](https://github.com/huggingface/diffusers/pull/11818) included
* You need to install `torchao` with [this fix](https://github.com/pytorch/ao/pull/2293) included

</details>

## Improvements, progressively
<details>
  <summary>Baseline</summary>

  For completeness, we demonstrate a (terrible) baseline here using the default `torch.float32` dtype.
  There's no practical reason do this over loading in `torch.bfloat16`, and the results are slow enough
  that they ruin the readability of the graph above when included (~7.5 sec).

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

</details>

<details>
  <summary>BFloat16</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

</details>

<details>
  <summary>torch.compile</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Compile the compute-intensive portions of the model: denoising transformer / decoder
# "max-autotune" mode tunes kernel hyperparameters and applies CUDAGraphs
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="max-autotune", fullgraph=True
)
pipeline.vae.decode = torch.compile(
    pipeline.vae.decode, mode="max-autotune", fullgraph=True
)

# warmup for a few iterations; trigger compilation
for _ in range(3):
    pipeline(
        "dummy prompt to trigger torch compilation",
        output_type="pil",
        num_inference_steps=4,
    ).images[0]

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

</details>

<details>
  <summary>Combining attention projection matrices</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Use channels_last memory format
pipeline.vae = pipeline.vae.to(memory_format=torch.channels_last)

# Combine attention projection matrices for (q, k, v)
pipeline.transformer.fuse_qkv_projections()
pipeline.vae.fuse_qkv_projections()

# compilation details omitted (see above)
...

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

Note that `torch.compile` is able to perform this fusion automatically, so we do not
observe a speedup from the fusion (outside of noise) when `torch.compile` is enabled.

</details>

<details>
  <summary>channels_last memory format</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Use channels_last memory format
pipeline.vae.to(memory_format=torch.channels_last)

# compilation details omitted (see above)
...

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

</details>

<details>
  <summary>Flash Attention V3 / aiter</summary>

  Flash Attention V3 is substantially faster on H100s than the previous iteration FA2, due
  in large part to float8 support. As this kernel isn't quite available yet within PyTorch Core, we implement a custom
  attention processor [`FlashFusedFluxAttnProcessor3_0`](./utils/pipeline_utils.py#L70) that uses the `flash_attn_interface`
  python bindings directly. We also ensure proper PyTorch custom op integration so that
  the op integrates well with `torch.compile` / `torch.export`. Inputs are converted to float8 in an unscaled fashion before
  kernel invocation and outputs are converted back to the original dtype on the way out.

  On AMD GPUs, we use [`aiter`](https://github.com/ROCm/aiter) instead, which also provides fp8 MHA kernels.

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Use channels_last memory format
pipeline.vae.to(memory_format=torch.channels_last)

# Combine attention projection matrices for (q, k, v)
pipeline.transformer.fuse_qkv_projections()
pipeline.vae.fuse_qkv_projections()

# Use FA3; reference FlashFusedFluxAttnProcessor3_0 impl for details
pipeline.transformer.set_attn_processor(FlashFusedFluxAttnProcessor3_0())

# compilation details omitted (see above)
...

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

</details>

<details>
  <summary>float8 quantization</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Use channels_last memory format
pipeline.vae.to(memory_format=torch.channels_last)

# Combine attention projection matrices for (q, k, v)
pipeline.transformer.fuse_qkv_projections()
pipeline.vae.fuse_qkv_projections()

# Use FA3; reference FlashFusedFluxAttnProcessor3_0 impl for details
pipeline.transformer.set_attn_processor(FlashFusedFluxAttnProcessor3_0())

# Apply float8 quantization on weights and activations
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight

quantize_(
    pipeline.transformer,
    float8_dynamic_activation_float8_weight(),
)

# compilation details omitted (see above)
...

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

</details>

<details>
  <summary>Inductor tuning flags</summary>

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Use channels_last memory format
pipeline.vae.to(memory_format=torch.channels_last)

# Combine attention projection matrices for (q, k, v)
pipeline.transformer.fuse_qkv_projections()
pipeline.vae.fuse_qkv_projections()

# Use FA3; reference FlashFusedFluxAttnProcessor3_0 impl for details
pipeline.transformer.set_attn_processor(FlashFusedFluxAttnProcessor3_0())

# Apply float8 quantization on weights and activations
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight

quantize_(
    pipeline.transformer,
    float8_dynamic_activation_float8_weight(),
)

# Tune Inductor flags
config = torch._inductor.config
config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
# adjust autotuning algorithm
config.coordinate_descent_tuning = True
config.coordinate_descent_check_all_directions = True
config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

# compilation details omitted (see above)
...

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

</details>

<details>
  <summary>torch.export + Ahead-Of-Time Inductor (AOTI)</summary>

  To avoid initial compilation times, we can use `torch.export` + Ahead-Of-Time Inductor (AOTI). This will
  serialize a binary, precompiled form of the model without initial compilation overhead.

```python
# Apply torch.export + AOTI. If serialize=True, writes out the exported models within the cache_dir.
# Otherwise, attempts to load previously-exported models from the cache_dir.
# This function also applies CUDAGraphs on the loaded models.
def use_export_aoti(pipeline, cache_dir, serialize=False):
    from torch._inductor.package import load_package

    # create cache dir if needed
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _example_tensor(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    # === Transformer export ===
    # torch.export requires a representative set of example args to be passed in
    transformer_kwargs = {
        "hidden_states": _example_tensor(1, 4096, 64),
        "timestep": torch.tensor([1.], device="cuda", dtype=torch.bfloat16),
        "guidance": None,
        "pooled_projections": _example_tensor(1, 768),
        "encoder_hidden_states": _example_tensor(1, 512, 4096),
        "txt_ids": _example_tensor(512, 3),
        "img_ids": _example_tensor(4096, 3),
        "joint_attention_kwargs": {},
        "return_dict": False,
    }

    # Possibly serialize model out
    transformer_package_path = os.path.join(cache_dir, "exported_transformer.pt2")
    if serialize:
        # Apply export
        exported_transformer: torch.export.ExportedProgram = torch.export.export(
            pipeline.transformer, args=(), kwargs=transformer_kwargs
        )

        # Apply AOTI
        path = torch._inductor.aoti_compile_and_package(
            exported_transformer,
            package_path=transformer_package_path,
            inductor_configs={"max_autotune": True, "triton.cudagraphs": True},
        )

    loaded_transformer = load_package(
        transformer_package_path, run_single_threaded=True
    )

    # warmup before cudagraphing
    with torch.no_grad():
        loaded_transformer(**transformer_kwargs)

    # Apply CUDAGraphs. CUDAGraphs are utilized in torch.compile with mode="max-autotune", but
    # they must be manually applied for torch.export + AOTI.
    loaded_transformer = cudagraph(loaded_transformer)
    pipeline.transformer.forward = loaded_transformer

    # warmup after cudagraphing
    with torch.no_grad():
        pipeline.transformer(**transformer_kwargs)

    # hack to get around export's limitations
    pipeline.vae.forward = pipeline.vae.decode

    vae_decode_kwargs = {
        "return_dict": False,
    }

    # Possibly serialize model out
    decoder_package_path = os.path.join(cache_dir, "exported_decoder.pt2")
    if serialize:
        # Apply export
        exported_decoder: torch.export.ExportedProgram = torch.export.export(
            pipeline.vae, args=(_example_tensor(1, 16, 128, 128),), kwargs=vae_decode_kwargs
        )

        # Apply AOTI
        path = torch._inductor.aoti_compile_and_package(
            exported_decoder,
            package_path=decoder_package_path,
            inductor_configs={"max_autotune": True, "triton.cudagraphs": True},
        )

    loaded_decoder = load_package(decoder_package_path, run_single_threaded=True)

    # warmup before cudagraphing
    with torch.no_grad():
        loaded_decoder(_example_tensor(1, 16, 128, 128), **vae_decode_kwargs)

    loaded_decoder = cudagraph(loaded_decoder)
    pipeline.vae.decode = loaded_decoder

    # warmup for a few iterations
    for _ in range(3):
        pipeline(
            "dummy prompt to trigger torch compilation",
            output_type="pil",
            num_inference_steps=4,
        ).images[0]

    return pipeline
```

Note that, unlike for `torch.compile`, running a model loaded from the torch.export + AOTI workflow
doesn't use CUDAGraphs by default. This was found to result in a ~5% performance decrease vs. torch.compile.
To address this discrepancy, we manually record / replay CUDAGraphs over the exported models using the following helper:
```python
# wrapper to automatically handle CUDAGraph record / replay over the given function
def cudagraph(f):
    from torch.utils._pytree import tree_map_only

    _graphs = {}
    def f_(*args, **kwargs):
        key = hash(tuple(tuple(kwargs[a].shape) for a in sorted(kwargs.keys())
                         if isinstance(kwargs[a], torch.Tensor)))
        if key in _graphs:
            # use the cached wrapper if one exists. this will perform CUDAGraph replay
            wrapped, *_ = _graphs[key]
            return wrapped(*args, **kwargs)

        # record a new CUDAGraph and cache it for future use
        g = torch.cuda.CUDAGraph()
        in_args, in_kwargs = tree_map_only(torch.Tensor, lambda t: t.clone(), (args, kwargs))
        f(*in_args, **in_kwargs) # stream warmup
        with torch.cuda.graph(g):
            out_tensors = f(*in_args, **in_kwargs)
        def wrapped(*args, **kwargs):
            # note that CUDAGraphs require inputs / outputs to be in fixed memory locations.
            # inputs must be copied into the fixed input memory locations.
            [a.copy_(b) for a, b in zip(in_args, args) if isinstance(a, torch.Tensor)]
            for key in kwargs:
                if isinstance(kwargs[key], torch.Tensor):
                    in_kwargs[key].copy_(kwargs[key])
            g.replay()
            # clone() outputs on the way out to disconnect them from the fixed output memory
            # locations. this allows for CUDAGraph reuse without accidentally overwriting memory
            return [o.clone() for o in out_tensors]

        # cache function that does CUDAGraph replay
        _graphs[key] = (wrapped, g, in_args, in_kwargs, out_tensors)
        return wrapped(*args, **kwargs)
    return f_
```

Finally, here is the fully-optimized form of the model:

```python
from diffusers import FluxPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell"
).to("cuda")

# Use channels_last memory format
pipeline.vae.to(memory_format=torch.channels_last)

# Combine attention projection matrices for (q, k, v)
pipeline.transformer.fuse_qkv_projections()
pipeline.vae.fuse_qkv_projections()

# Use FA3; reference FlashFusedFluxAttnProcessor3_0 impl for details
pipeline.transformer.set_attn_processor(FlashFusedFluxAttnProcessor3_0())

# Apply float8 quantization on weights and activations
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight

quantize_(
    pipeline.transformer,
    float8_dynamic_activation_float8_weight(),
)

# Tune Inductor flags
config = torch._inductor.config
config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
# adjust autotuning algorithm
config.coordinate_descent_tuning = True
config.coordinate_descent_check_all_directions = True
config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

# Apply torch.export + AOTI with CUDAGraphs
pipeline = use_export_aoti(pipeline, cache_dir=args.cache_dir, serialize=False)

prompt = "A cat playing with a ball of yarn"
image = pipe(prompt, num_inference_steps=4).images[0]
```

</details>


<details>
  <summary>Cache Acceleration with cache-dit: DBCache + F12B12</summary>

You can use `cache-dit` to further speedup FLUX model. For example:

```python
# cache-dit: DBCache F12B12
# install: pip install -U cache-dit
from diffusers import FluxPipeline
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Custom options, F8B16, higher precision
cache_options = {
    "cache_type": CacheType.DBCache,
    "warmup_steps": 8,
    "max_cached_steps": 8,    # -1 means no limit
    "Fn_compute_blocks": 12,  # Fn, F12, etc.
    "Bn_compute_blocks": 12,  # Bn, B12, etc.
    "residual_diff_threshold": 0.12,
}

apply_cache_on_pipe(pipeline, **cache_options)
```

By the way, `cache-dit` is designed to work compatibly with torch.compile. You can easily use `cache-dit` with torch.compile to further achieve a better performance. For example:

```python
cache_options = {
    "cache_type": CacheType.DBCache,
    "warmup_steps": 8,
    "max_cached_steps": 8,    # -1 means no limit
    "Fn_compute_blocks": 12,  # Fn, F12, etc.
    "Bn_compute_blocks": 12,  # Bn, B12, etc.
    "residual_diff_threshold": 0.12,
}

apply_cache_on_pipe(pipeline, **cache_options)

# Compile the Transformer module
pipeline.transformer = torch.compile(pipeline.transformer)
```
However, users intending to use `cache-dit` for DiT with dynamic input shapes should consider increasing the recompile limit of torch._dynamo. Otherwise, the recompile_limit error may be triggered, causing the module to fall back to eager mode.

```python
torch._dynamo.config.recompile_limit = 96  # default is 8
torch._dynamo.config.accumulated_recompile_limit = 2048  # default is 256
```

</details>