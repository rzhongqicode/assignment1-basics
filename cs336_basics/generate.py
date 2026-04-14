import argparse

import torch

from cs336_basics.building_blocks import AdamW, Transformer_lm, load_checkpoint, softmax
from cs336_basics.tokenizer import Tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 1.0,
    max_generate_length: int = 100,
    context_length: int = 512,
    top_p=1.0,
) -> str:
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokens = tokenizer.encode(prompt)
    initial_len = len(tokens)

    terminator = "<|endoftext|>"
    terminator_id = tokenizer.encode(terminator)[0]

    generated_token_num = 0
    while generated_token_num < max_generate_length:
        if len(tokens) < context_length:
            input_list = tokens
        else:
            input_list = tokens[-context_length:]

        # add [] around input_list to add dimension to [1, seq_len]
        input = torch.tensor(data=[input_list], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input)

        next_token_logits = logits[0, -1, :]
        if temperature == 0:
            sampled_id = torch.argmax(next_token_logits).item()
        else:
            next_token_logits = next_token_logits / temperature
            next_token_distribution = softmax(in_features=next_token_logits, dim=-1)
            # top-p sampling
            if top_p < 1.0:
                # sort the probs
                sorted_probs, sorted_indices = torch.sort(next_token_distribution, descending=True)
                # calculate cumulative probs
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p

                # shift right 1 bit
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                # remove probs
                sorted_probs[sorted_indices_to_remove] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
                next_token_idx = torch.multinomial(input=sorted_probs, num_samples=1)
                sampled_id = sorted_indices[next_token_idx].item()

            else:
                sampled_id = torch.multinomial(input=next_token_distribution, num_samples=1).item()

        if sampled_id == terminator_id:
            break
        else:
            tokens.append(sampled_id)
            generated_token_num += 1

    generated_str = tokenizer.decode(tokens[initial_len:])
    return generated_str


def main():
    parser = argparse.ArgumentParser(
        description="input model_path, vocab_path, merges_path and prompt to generate a text"
    )
    parser.add_argument("model_path", help="model path")
    parser.add_argument("vocab_path", help="vocab_path")
    parser.add_argument("merges_path", help="merges_path")
    parser.add_argument("prompt", help="prompt")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("-m", "--max_generate_length", type=int, default=100, help="maximum of generated text length")
    parser.add_argument("-c", "--context_length", type=int, default=512, help="context length of the model")
    parser.add_argument("-p", "--top_p", type=float, default=1.0, help="top p")

    args = parser.parse_args()
    model_path = args.model_path
    vocab_path = args.vocab_path
    merges_path = args.merges_path
    prompt = args.prompt
    temperature = args.temperature
    max_generate_length = args.max_generate_length
    context_length = args.context_length
    top_p = args.top_p

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer_lm(
        vocab_size=10000,
        context_length=context_length,
        num_layers=12,
        num_heads=8,
        d_model=512,
        d_ff=int(8 / 3 * 512),
        rope_theta=10000,
        device=device,
    )
    model.to(device)
    # weight_decay = 0.05
    # optimizer = AdamW(params=model.parameters(), lr=0.0, weight_decay=weight_decay)
    # t = load_checkpoint(src=model_path, model=model, optimizer=optimizer)
    checkpoint_dict = torch.load(f=model_path, map_location=device)
    model.load_state_dict(checkpoint_dict["model"])
    print(f"model load successfully! model path: {model_path}")

    # load tokenizer
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=special_tokens
    )

    # generate response
    print("generating... please wait~")
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temperature=temperature,
        max_generate_length=max_generate_length,
        context_length=context_length,
        top_p=top_p,
    )

    print(f"response: {output}")


if __name__ == "__main__":
    main()
