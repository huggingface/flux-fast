import random
import torch
from utils.benchmark_utils import create_parser
from utils.pipeline_utils import load_pipeline  # noqa: E402
from run_benchmark import _determine_pipe_call_kwargs

def set_rand_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    pipeline = load_pipeline(args)
    set_rand_seeds(args.seed)

    image = pipeline(
        prompt=args.prompt, 
        num_inference_steps=args.num_inference_steps, 
        generator=torch.manual_seed(args.seed),
        **_determine_pipe_call_kwargs(args)
    ).images[0]
    image.save(args.output_file)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
