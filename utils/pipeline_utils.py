import os
import pathlib
import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from torch._inductor.package import load_package
from typing import List, Optional, Tuple
from PIL import Image


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

    import flash_attn_interface

    dtype = torch.float8_e4m3fn
    outputs = flash_attn_interface.flash_attn_func(
        q.to(dtype),
        k.to(dtype),
        v.to(dtype),
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        # sink_token_length=sink_token_length, unknown keyword argument 🤔
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
        sm_margin=sm_margin,
    )
    return outputs[0]


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
        try:
            import flash_attn_interface
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


def cudagraph(f):
    from torch.utils._pytree import tree_map_only

    _graphs = {}
    def f_(*args, **kwargs):
        key = hash(tuple(tuple(kwargs[a].shape) for a in sorted(kwargs.keys())
                         if isinstance(kwargs[a], torch.Tensor)))
        if key in _graphs:
            wrapped, *_ = _graphs[key]
            return wrapped(*args, **kwargs)
        g = torch.cuda.CUDAGraph()
        in_args, in_kwargs = tree_map_only(torch.Tensor, lambda t: t.clone(), (args, kwargs))
        f(*in_args, **in_kwargs) # stream warmup
        with torch.cuda.graph(g):
            out_tensors = f(*in_args, **in_kwargs)
        def wrapped(*args, **kwargs):
            [a.copy_(b) for a, b in zip(in_args, args) if isinstance(a, torch.Tensor)]
            for key in kwargs:
                if isinstance(kwargs[key], torch.Tensor):
                    in_kwargs[key].copy_(kwargs[key])
            g.replay()
            return [o.clone() for o in out_tensors]
        _graphs[key] = (wrapped, g, in_args, in_kwargs, out_tensors)
        return wrapped(*args, **kwargs)
    return f_


def use_compile(pipeline):
    # Compile the compute-intensive portions of the model: denoising transformer / decoder
    pipeline.transformer = torch.compile(
        pipeline.transformer, mode="max-autotune", fullgraph=True
    )
    pipeline.vae.decode = torch.compile(
        pipeline.vae.decode, mode="max-autotune", fullgraph=True
    )

    # warmup for a few iterations
    for _ in range(3):
        pipeline(
            "dummy prompt to trigger torch compilation",
            output_type="pil",
            num_inference_steps=4,
        ).images[0]

    return pipeline


def download_hosted_file(filename, output_path):
    # Download hosted binaries from huggingface Hub.
    from huggingface_hub import hf_hub_download

    REPO_NAME = "jbschlosser/flux-fast"
    hf_hub_download(REPO_NAME, filename, local_dir=os.path.dirname(output_path))


def use_export_aoti(pipeline, cache_dir, serialize=False):
    def _example_tensor(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    is_kontext = pipeline.__class__.__name__ == "FluxKontextPipeline"

    # === Transformer compile / export ===
    transformer_kwargs = {
        "hidden_states": _example_tensor(1, 8137, 64) if is_kontext else _example_tensor(1, 4096, 64),
        "timestep": torch.tensor([1.], device="cuda", dtype=torch.bfloat16) / 1000,
        "guidance": torch.tensor([2.5], device="cuda", dtype=torch.bfloat16) if is_kontext else None,
        "pooled_projections": _example_tensor(1, 768),
        "encoder_hidden_states": _example_tensor(1, 512, 4096),
        "txt_ids": _example_tensor(512, 3),
        "img_ids": _example_tensor(8137, 3) if is_kontext else _example_tensor(4096, 3),
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
    # download serialized model if needed
    if not os.path.exists(transformer_package_path):
        download_hosted_file(os.path.basename(transformer_package_path), transformer_package_path)

    loaded_transformer = load_package(
        transformer_package_path, run_single_threaded=True
    )

    # warmup before cudagraphing
    with torch.no_grad():
        loaded_transformer(**transformer_kwargs)

    # Apply CUDAGraphs
    loaded_transformer = cudagraph(loaded_transformer)
    pipeline.transformer.forward = loaded_transformer

    # warmup after cudagraping
    with torch.no_grad():
        pipeline.transformer(**transformer_kwargs)

    # hack to get around export's limitations
    pipeline.vae.forward = pipeline.vae.decode

    vae_decode_kwargs = {
        "return_dict": False,
    }

    # Possibly serialize model out
    decoder_package_path = os.path.join(cache_dir, "exported_decoder.pt2")
    decoder_args = _example_tensor(1, 16, 106, 154) if is_kontext else _example_tensor(1, 16, 128, 128)
    if serialize:
        # Apply export
        exported_decoder: torch.export.ExportedProgram = torch.export.export(
            pipeline.vae, args=(decoder_args,), kwargs=vae_decode_kwargs
        )

        # Apply AOTI
        path = torch._inductor.aoti_compile_and_package(
            exported_decoder,
            package_path=decoder_package_path,
            inductor_configs={"max_autotune": True, "triton.cudagraphs": True},
        )
    # download serialized model if needed
    if not os.path.exists(decoder_package_path):
        download_hosted_file(os.path.basename(decoder_package_path), decoder_package_path)

    loaded_decoder = load_package(decoder_package_path, run_single_threaded=True)

    # warmup before cudagraphing
    with torch.no_grad():
        loaded_decoder(decoder_args, **vae_decode_kwargs)

    loaded_decoder = cudagraph(loaded_decoder)
    pipeline.vae.decode = loaded_decoder

    # warmup for a few iterations
    pipe_kwargs = {
        "prompt": "dummy prompt to trigger torch compilation",
        "output_type": "pil",
        "num_inference_steps": 4
    }
    if is_kontext:
        pipe_kwargs.update(
            {
                "image": Image.new("RGB", size=(1024, 704)), # special shape for now.
                "height": 704,
                "width": 1024
            }, 
        )
    for _ in range(3):
        pipeline(**pipe_kwargs).images[0]

    return pipeline


# If lossy=False, only lossless optimizations are performed
def optimize(pipeline, cache_dir, lossy=True):
    pipeline.set_progress_bar_config(disable=True)

    # fuse QKV projections in Transformer and VAE
    pipeline.transformer.fuse_qkv_projections()
    pipeline.vae.fuse_qkv_projections()

    # Use flash attention v3
    pipeline.transformer.set_attn_processor(FlashFusedFluxAttnProcessor3_0())

    # switch memory layout to Torch's preferred, channels_last
    pipeline.transformer.to(memory_format=torch.channels_last)
    pipeline.vae.to(memory_format=torch.channels_last)

    if lossy:
        # apply float8 quantization
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, PerRow

        quantize_(
            pipeline.transformer,
            float8_dynamic_activation_float8_weight(granularity=PerRow()),
        )
        quantize_(
            pipeline.vae,
            float8_dynamic_activation_float8_weight(granularity=PerRow()),
        )

    # set inductor flags
    config = torch._inductor.config
    config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
    # adjust autotuning algorithm
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

    # TODO: Mess around more with mm settings
    # config.triton.enable_persistent_tma_matmul = True
    # config.max_autotune_gemm_backends = "ATEN,TRITON,CPP,CUTLASS"

    # pipeline = use_compile(pipeline)
    # NB: Using a cached export + AOTI model is not supported yet
    pipeline = use_export_aoti(pipeline, cache_dir=cache_dir, serialize=True)

    return pipeline


def load_pipeline(options):
    # create cache dir if needed
    cache_dir = options.get("cache_dir", os.path.expandvars("$HOME/.cache/flux-fast"))
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # schnell
    benchmark_schnell = options.get("benchmark_schnell", True)
    benchmark_kontext = options.get("benchmark_kontext", False)
    if benchmark_schnell:
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        ).to("cuda")
    elif benchmark_kontext:
        from kontext_pipeline import FluxKontextPipeline
        from diffusers import FluxTransformer2DModel
        from huggingface_hub import hf_hub_download

        kontext_path = hf_hub_download(repo_id="diffusers/kontext", filename="kontext.safetensors")
        transformer = FluxTransformer2DModel.from_single_file(kontext_path, torch_dtype=torch.bfloat16)
        pipeline = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
        ).to("cuda")

    enable_optims = options.get("enable_optims", False)
    if enable_optims:
        pipeline = optimize(pipeline, cache_dir=cache_dir, lossy=True)

    return pipeline
