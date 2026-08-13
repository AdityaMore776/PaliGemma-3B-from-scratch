import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image

from model import KVCache, PaliGemmaProcessor, load_hf_model

ROOT = Path(__file__).resolve().parent
MODEL_ID = "google/paligemma-3b-mix-224"
MODEL_DIR = ROOT / "models" / "paligemma-3b-mix-224"


def get_device(only_cpu=False):
    if only_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_model():
    if (MODEL_DIR / "config.json").exists():
        return MODEL_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_DIR))
    return MODEL_DIR


def load_pipeline(device):
    model_path = ensure_model()
    model, tokenizer = load_hf_model(str(model_path), device)
    model.eval()
    processor = PaliGemmaProcessor(
        tokenizer,
        model.config.vision_config.num_image_tokens,
        model.config.vision_config.image_size,
    )
    return model, processor, device


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


@torch.no_grad()
def generate(model, processor, device, prompt, image_path, max_tokens=100, temperature=0.8, top_p=0.9, do_sample=False):
    image = Image.open(image_path).convert("RGB")
    model_inputs = {k: v.to(device) for k, v in processor(text=[prompt], images=[image]).items()}
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    kv_cache = KVCache()
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    for _ in range(max_tokens):
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)
        if next_token.item() == stop_token:
            break
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=-1)
    if not generated_tokens:
        return prompt
    decoded = processor.tokenizer.decode(torch.cat(generated_tokens, dim=-1), skip_special_tokens=True)
    return prompt + decoded


def run_once(model, processor, device, args):
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    result = generate(
        model,
        processor,
        device,
        args.prompt,
        str(image_path),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.sample,
    )
    print(result)


def run_loop(model, processor, device, args):
    last_image = args.image
    print("Enter image path and prompt each run. Type q to quit.\n")
    while True:
        image_input = input(f"image [{last_image}]: ").strip()
        if image_input.lower() in ("q", "quit", "exit"):
            break
        prompt = input("prompt: ").strip()
        if prompt.lower() in ("q", "quit", "exit"):
            break
        if not prompt:
            continue
        image_path = image_input or last_image
        last_image = image_path
        args.image = image_path
        args.prompt = prompt
        try:
            run_once(model, processor, device, args)
        except FileNotFoundError:
            print(f"not found: {image_path}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="Describe this image.")
    parser.add_argument("image", nargs="?", default="images/screenshot.png")
    parser.add_argument("-p", "--prompt", dest="prompt_flag", default=None)
    parser.add_argument("-i", "--image", dest="image_flag", default=None)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    if args.prompt_flag:
        args.prompt = args.prompt_flag
    if args.image_flag:
        args.image = args.image_flag
    device = get_device(args.cpu)
    model, processor, device = load_pipeline(device)
    if args.loop:
        run_loop(model, processor, device, args)
    else:
        run_once(model, processor, device, args)


if __name__ == "__main__":
    main()
