#!/bin/bash

CKPT="black-forest-labs/FLUX.1-Kontext-dev"
IMAGE="yarn-art-pikachu.png"
PROMPT="Make Pikachu hold a sign that says 'Black Forest Labs is awesome', yarn art style, detailed, vibrant colors"
CACHE_DIR="/fsx/sayak/.cache"

# bfloat16
python run_benchmark.py \
    --ckpt $CKPT --image $IMAGE --prompt "$PROMPT" \
    --compile_export_mode disabled \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --output-file bf16.png \
    --num_inference_steps 28 \
    --disable_cache_dit \
    --cache-dir $CACHE_DIR \
    > bf16.txt 2>&1

# bfloat16 + torch.compile
python run_benchmark.py \
    --ckpt $CKPT --image $IMAGE --prompt "$PROMPT" \
    --compile_export_mode compile \
    --disable_fused_projections \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --output-file bf16_compile.png \
    --num_inference_steps 28 \
    --disable_cache_dit \
    --cache-dir $CACHE_DIR \
    > bf16_compile.txt 2>&1

# bfloat16 + torch.compile + qkv projection
python run_benchmark.py \
    --ckpt $CKPT --image $IMAGE --prompt "$PROMPT" \
    --compile_export_mode compile \
    --disable_channels_last \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --output-file bf16_compile_qkv.png \
    --num_inference_steps 28 \
    --disable_cache_dit \
    --cache-dir $CACHE_DIR \
    > bf16_compile_qkv.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last
python run_benchmark.py \
    --ckpt $CKPT --image $IMAGE --prompt "$PROMPT" \
    --compile_export_mode compile \
    --disable_fa3 \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --output-file bf16_compile_qkv_chan.png \
    --num_inference_steps 28 \
    --disable_cache_dit \
    --cache-dir $CACHE_DIR \
    > bf16_compile_qkv_chan.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + FA3
python run_benchmark.py \
    --ckpt $CKPT --image $IMAGE --prompt "$PROMPT" \
    --compile_export_mode compile \
    --disable_quant \
    --disable_inductor_tuning_flags \
    --output-file bf16_compile_qkv_chan_fa3.png \
    --num_inference_steps 28 \
    --disable_cache_dit \
    --cache-dir $CACHE_DIR \
    > bf16_compile_qkv_chan_fa3.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + FA3 + float8 quant
python run_benchmark.py \
    --ckpt $CKPT --image $IMAGE --prompt "$PROMPT" \
    --compile_export_mode compile \
    --disable_inductor_tuning_flags \
    --output-file bf16_compile_qkv_chan_fa3_quant.png \
    --num_inference_steps 28 \
    --disable_cache_dit \
    --cache-dir $CACHE_DIR \
    > bf16_compile_qkv_chan_fa3_quant.txt 2>&1

# bfloat16 + torch.compile + qkv projection + channels_last + FA3 + float8 quant + inductor flags
python run_benchmark.py \
    --ckpt $CKPT --image $IMAGE --prompt "$PROMPT" \
    --compile_export_mode compile \
    --output-file bf16_compile_qkv_chan_fa3_quant_flags.png \
    --num_inference_steps 28 \
    --disable_cache_dit \
    --cache-dir $CACHE_DIR \
    > bf16_compile_qkv_chan_fa3_quant_flags.txt 2>&1

# fully optimized (torch.export + AOTI to address cold start)
python run_benchmark.py --ckpt $CKPT --image $IMAGE --prompt "$PROMPT" \
    --output-file fully_optimized.png \
    --num_inference_steps 28 \
    --disable_cache_dit \
    --cache-dir $CACHE_DIR \
    > fully_optimized.txt 2>&1
