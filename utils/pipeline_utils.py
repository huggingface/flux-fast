import os
import pathlib
import torch
from diffusers import DiffusionPipeline
from torch._inductor.package import load_package as inductor_load_package
from typing import List, Optional
from PIL import Image
import inspect

def is_hip():
    return torch.version.hip is not None


@torch.library.custom_op("flash::flash_attn_func", mutates_args=())
def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] =None,
    causal: bool = False,
    # probably wrong type for these 4
    qv: Optional[float] = None,
    q_descale: Optional[float] = None,
    k_descale: Optional[float] = None,
    v_descale: Optional[float] = None,
    window_size: Optional[List[int]] = None,
    sink_token_length: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    # probably wrong type for this too
    pack_gqa: Optional[float] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> torch.Tensor: #Tuple[torch.Tensor, torch.Tensor]:
    if window_size is None:
        window_size = (-1, -1)
    else:
        window_size = tuple(window_size)

    if is_hip():
        from aiter.ops.triton.mha import flash_attn_fp8_func as flash_attn_interface_func
    else:
        from flash_attn.flash_attn_interface import flash_attn_interface_func

    sig = inspect.signature(flash_attn_interface_func)
    accepted = set(sig.parameters)
    all_kwargs = {
        "softmax_scale": softmax_scale,
        "causal": causal,
        "qv": qv,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "window_size": window_size,
        "sink_token_length": sink_token_length,
        "softcap": softcap,
        "num_splits": num_splits,
        "pack_gqa": pack_gqa,
        "deterministic": deterministic,
        "sm_margin": sm_margin,
    }
    kwargs = {k: v for k, v in all_kwargs.items() if k in accepted}

    if is_hip():
        # For AMD, AITER fp8 kernels take in fp32 inputs and converts it to fp8 by itself
        # So we don't need to convert to fp8 here
        outputs = flash_attn_interface_func(
            q, k, v, **kwargs,
        )
    else:
        dtype = torch.float8_e4m3fn
        outputs = flash_attn_interface_func(
            q.to(dtype), k.to(dtype), v.to(dtype), **kwargs,
        )[0]

    return outputs.contiguous().to(torch.bfloat16) if is_hip() else outputs

@flash_attn_func.register_fake
def _(q, k, v, **kwargs):
    # two outputs:
    # 1. output: (batch, seq_len, num_heads, head_dim)
    # 2. softmax_lse: (batch, num_heads, seq_len) with dtype=torch.float32
    meta_q = torch.empty_like(q).contiguous()
    return meta_q #, q.new_empty((q.size(0), q.size(2), q.size(1)), dtype=torch.float32)

# Copied FusedFluxAttnProcessor2_0 but using flash v3 instead of SDPA
class FlashFusedFluxAttnProcessor3_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):

        if is_hip():
            try:
                from aiter.ops.triton.mha import flash_attn_fp8_func as flash_attn_interface_func
            except ImportError:
                raise ImportError(
                    "aiter is required to be installed"
                )
        else:
            try:
                from flash_attn.flash_attn_interface import flash_attn_interface_func
            except ImportError:
                raise ImportError(
                    "flash_attention v3 package is required to be installed"
                )

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = torch.split(qkv, split_size, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
            split_size = encoder_qkv.shape[-1] // 3
            (
                encoder_hidden_states_query_proj,
                encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj,
            ) = torch.split(encoder_qkv, split_size, dim=-1)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # NB: transposes are necessary to match expected SDPA input shape
        hidden_states = flash_attn_func(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2))[0].transpose(1, 2)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


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


def use_compile(pipeline):
    # Compile the compute-intensive portions of the model: denoising transformer / decoder
    is_kontext = "Kontext" in pipeline.__class__.__name__
    # Compile transformer w/o fullgraph and cudagraphs if cache-dit is enabled.
    # The cache-dit relies heavily on dynamic Python operations to maintain the cache_context, 
    # so it is necessary to introduce graph breaks at appropriate positions to be compatible 
    # with torch.compile. Thus, we compile the transformer with `max-autotune-no-cudagraphs` 
    # mode if cache-dit is enabled. Otherwise, we compile with `max-autotune` mode.
    is_cached = getattr(pipeline.transformer, "_is_cached", False)
    # For AMD MI300X w/ the AITER kernels, the default dynamic=None is not working as expected, giving black results.
    # Therefore, we use dynamic=True for AMD only. This leads to a small perf penalty, but should be fixed eventually. 
    pipeline.transformer = torch.compile(
        pipeline.transformer, 
        mode="max-autotune" if not is_cached else "max-autotune-no-cudagraphs", 
        fullgraph=(True if not is_cached else False), 
        dynamic=True if is_hip() else None
    )
    pipeline.vae.decode = torch.compile(
        pipeline.vae.decode, mode="max-autotune", fullgraph=True, dynamic=True if is_hip() else None
    )

    # warmup for a few iterations (`num_inference_steps` shouldn't matter)
    input_kwargs = {
        "prompt": "dummy prompt to trigger torch compilation", "num_inference_steps": 4
    }
    if is_kontext:
        input_kwargs.update({"image": Image.new("RGB", size=(1024, 1024))})
    for _ in range(3):
        pipeline(**input_kwargs).images[0]

    return pipeline


def download_hosted_file(filename, output_path):
    # Download hosted binaries from huggingface Hub.
    from huggingface_hub import hf_hub_download

    REPO_NAME = "jbschlosser/flux-fast"
    hf_hub_download(REPO_NAME, filename, local_dir=os.path.dirname(output_path))


def load_package(package_path):
    if not os.path.exists(package_path):
        download_hosted_file(os.path.basename(package_path), package_path)

    loaded_package = inductor_load_package(package_path, run_single_threaded=True)
    return loaded_package


def use_export_aoti(pipeline, cache_dir, serialize=False, is_timestep_distilled=True):
    # create cache dir if needed
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _example_tensor(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    # helpful flag
    is_kontext = "Kontext" in pipeline.__class__.__name__

    # === Transformer compile / export ===
    seq_length = 256 if is_timestep_distilled else 512
    # these shapes are for 1024x1024 resolution.
    transformer_kwargs = {
        "hidden_states": _example_tensor(1, 4096 * 2, 64) if is_kontext else _example_tensor(1, 4096, 64),
        "timestep": torch.tensor([1.], device="cuda", dtype=torch.bfloat16),
        "guidance": None if is_timestep_distilled else torch.tensor([1.], device="cuda", dtype=torch.bfloat16),
        "pooled_projections": _example_tensor(1, 768),
        "encoder_hidden_states": _example_tensor(1, seq_length, 4096),
        "txt_ids": _example_tensor(seq_length, 3),
        "img_ids": _example_tensor(4096 * 2, 3) if is_kontext else _example_tensor(4096, 3),
        "joint_attention_kwargs": {},
        "return_dict": False,
    }

    # Possibly serialize model out
    dev_transformer_name = "exported_kontext_dev_transformer.pt2" if is_kontext else "exported_dev_transformer.pt2"
    transformer_package_path = os.path.join(
        cache_dir, "exported_transformer.pt2" if is_timestep_distilled else dev_transformer_name
    )
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
    # download serialized model if needed
    loaded_transformer = load_package(transformer_package_path)

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

    vae_decode_kwargs = {"return_dict": False}

    # Possibly serialize model out
    decoder_package_path = os.path.join(
        cache_dir, "exported_decoder.pt2" if is_timestep_distilled else "exported_dev_decoder.pt2"
    )
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
    # download serialized model if needed
    loaded_decoder = load_package(decoder_package_path)

    # warmup before cudagraphing
    with torch.no_grad():
        loaded_decoder(_example_tensor(1, 16, 128, 128), **vae_decode_kwargs)

    loaded_decoder = cudagraph(loaded_decoder)
    pipeline.vae.decode = loaded_decoder

    # warmup for a few iterations
    input_kwargs = {
        "prompt": "dummy prompt to trigger torch compilation", "num_inference_steps": 4
    }
    if is_kontext:
        input_kwargs.update({"image": Image.new("RGB", size=(1024, 1024))})
    for _ in range(3):
        pipeline(**input_kwargs).images[0]

    return pipeline


def optimize(pipeline, args):
    is_timestep_distilled = not pipeline.transformer.config.guidance_embeds

    # fuse QKV projections in Transformer and VAE
    if not args.disable_fused_projections:
        pipeline.transformer.fuse_qkv_projections()
        pipeline.vae.fuse_qkv_projections()

    # Use flash attention v3
    if not args.disable_fa3:
        pipeline.transformer.set_attn_processor(FlashFusedFluxAttnProcessor3_0())

    # switch memory layout to channels_last
    if not args.disable_channels_last:
        pipeline.vae.to(memory_format=torch.channels_last)
    
    # cache-dit: DBCache configs
    if args.cache_dit_config is not None:
        if args.ckpt.endswith("schnell"):
            raise ValueError(
                "cache-dit is not suitable for FLUX.1-schnell with only 4 steps "
                "(there's no need to use cache either), please try FLUX.1-dev "
                "or FLUX.1-Kontext-dev with 28 steps."
            )
        try:
            # docs: https://github.com/vipshop/cache-dit
            from cache_dit.cache_factory import apply_cache_on_pipe
            from cache_dit.cache_factory import load_cache_options_from_yaml
            cache_options = load_cache_options_from_yaml(
                args.cache_dit_config
            )
            apply_cache_on_pipe(pipeline, **cache_options)
        except ImportError as e:
            print(
                "You have passed the '--cache_dit_config' flag, but we cannot "
                "find 'cache-dit' in your environment. Please install it via "
                "'pip install -U cache-dit' first, or re-run the process without "
                "the '--cache_dit_config' flag."
            )
            raise e

    # apply float8 quantization
    if not args.disable_quant:
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight #, PerRow

        quantize_(
            pipeline.transformer,
            float8_dynamic_activation_float8_weight(),
            # float8_dynamic_activation_float8_weight(granularity=PerRow()),
        )

    # set inductor flags
    if not args.disable_inductor_tuning_flags:
        config = torch._inductor.config
        config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
        config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls
        # adjust autotuning algorithm
        config.coordinate_descent_tuning = True
        config.coordinate_descent_check_all_directions = True

        # TODO: Test out more mm settings
        # config.triton.enable_persistent_tma_matmul = True
        # config.max_autotune_gemm_backends = "ATEN,TRITON,CPP,CUTLASS"

    if args.compile_export_mode == "compile":
        pipeline = use_compile(pipeline)
    elif args.compile_export_mode == "export_aoti":
        if not args.enable_cache_dit:
            pipeline = use_export_aoti(
                pipeline,
                cache_dir=args.cache_dir,
                serialize=(not args.use_cached_model),
                is_timestep_distilled=is_timestep_distilled
            )
        else:
            print(
                "Currently, 'cache-dit' is incompatible with 'export_aoti'. "
                "Please disable 'cache-dit' and re-run the export process."
            )
    elif args.compile_export_mode == "disabled":
        pass
    else:
        raise RuntimeError(
            "expected compile_export_mode arg to be one of {compile, export_aoti, disabled}"
        )

    return pipeline


def load_pipeline(args):
    load_dtype = torch.float32 if args.disable_bf16 else torch.bfloat16
    pipeline = DiffusionPipeline.from_pretrained(args.ckpt, torch_dtype=load_dtype).to(args.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline = optimize(pipeline, args)
    return pipeline
