import random
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from utils.benchmark_utils import annotate, create_parser
from utils.pipeline_utils import load_pipeline  # noqa: E402
from diffusers.utils import load_image
import os

def _determine_pipe_call_kwargs(args):
    kwargs = {"max_sequence_length": 256, "guidance_scale": 0.0}
    ckpt_id = args.ckpt
    if ckpt_id == "black-forest-labs/FLUX.1-dev":
        kwargs = {"max_sequence_length": 512, "guidance_scale": 3.5}
    elif ckpt_id == "black-forest-labs/FLUX.1-Kontext-dev":
        kwargs = {"max_sequence_length": 512, "guidance_scale": 2.5}
        kwargs.update({"image": load_image(args.image)}) 
    return kwargs

def set_rand_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    set_rand_seeds(args.seed)
    pipeline = load_pipeline(args)
    if args.ckpt == "black-forest-labs/FLUX.1-Kontext-dev":
        assert os.path.exists(args.image)
    set_rand_seeds(args.seed)

    # warmup
    for _ in range(3):
        image = pipeline(
            prompt=args.prompt, 
            num_inference_steps=args.num_inference_steps, 
            generator=torch.manual_seed(args.seed),
            **_determine_pipe_call_kwargs(args)
        ).images[0]

    # run inference 10 times and compute mean / variance
    timings = []
    for _ in range(10):
        begin = time.time()
        image = pipeline(
            prompt=args.prompt, 
            num_inference_steps=args.num_inference_steps, 
            generator=torch.manual_seed(args.seed),
            **_determine_pipe_call_kwargs(args)
        ).images[0]
        end = time.time()
        timings.append(end - begin)
    timings = torch.tensor(timings)
    print('time mean/var:', timings, timings.mean().item(), timings.var().item())
    image.save(args.output_file)

    # optionally generate PyTorch Profiler trace
    # this is done after benchmarking because tracing introduces overhead
    if args.trace_file is not None:
        # annotate parts of the model within the profiler trace
        pipeline.transformer.forward = annotate(pipeline.transformer.forward, "denoising_step")
        pipeline.vae.decode = annotate(pipeline.vae.decode, "decoding")
        pipeline.encode_prompt = annotate(pipeline.encode_prompt, "prompt_encoding")
        pipeline.image_processor.postprocess = annotate(
            pipeline.image_processor.postprocess, "postprocessing"
        )
        pipeline.image_processor.numpy_to_pil = annotate(
            pipeline.image_processor.numpy_to_pil, "pil_conversion"
        )

        # Generate trace with the PyTorch Profiler
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("timed_region"):
                image = pipeline(
                    args.prompt, 
                    num_inference_steps=args.num_inference_steps, 
                    **_determine_pipe_call_kwargs(args)
                ).images[0]
        prof.export_chrome_trace(args.trace_file)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
