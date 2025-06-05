import random
import time
import torch
from PIL import Image
from utils.pipeline_utils import load_pipeline  # noqa: E402


# TODO: Update this to match diffusion-fast, make things more configurable via arg parser, etc.
def main():
    pipeline = load_pipeline(
        {
            "cache_dir": "/fsx/sayak/.cache", 
            "benchmark_kontext": True, 
            "benchmark_schnell": False,
            "enable_optims": True
        }
    )

    # get inputs.
    prompt = "Turn the image into a ghibli style image"
    image = Image.open("pexels-jmark-253096.png")
    original_width, original_height = image.size
    
    if original_width >= original_height:
        new_width = 1024
        new_height = int(original_height * (new_width / original_width))
        new_height = round(new_height / 64) * 64
    else:
        new_height = 1024
        new_width = int(original_width * (new_height / original_height))
        new_width = round(new_width / 64) * 64
    print(f"Resizing to: {new_height}x{new_width}")
    image_resized = image.resize((new_width, new_height), Image.LANCZOS)
    image_resized.save("image_resized.png")

    # warmup
    for _ in range(3):
        image = pipeline(
            prompt=prompt, image=image_resized, guidance_scale=2.5, height=new_height, width=new_width
        ).images[0]

    # run inference 10 times and compute mean / variance
    timings = []
    for _ in range(10):
        begin = time.time()
        image = pipeline(
            prompt=prompt, 
            image=image_resized, 
            guidance_scale=2.5, 
            height=new_height, 
            width=new_width,
            generator=torch.manual_seed(0)
        ).images[0]
        end = time.time()
        timings.append(end - begin)
    timings = torch.tensor(timings)
    print('time mean/var:', timings, timings.mean().item(), timings.var().item())
    image.save("output.png")


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    main()
