import argparse
import os

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from cs336_basics.building_blocks import (
    AdamW,
    Transformer_lm,
    cross_entropy,
    data_loader,
    gradient_clipping,
    learning_rate_schedule,
    save_checkpoint,
)


def main():
    # make dir to save the model
    os.makedirs(name="./checkpoint", exist_ok=True)

    # parse the arguments
    parser = argparse.ArgumentParser(description="input all the arguments to train the model")
    parser.add_argument("training", help="training data file name")
    parser.add_argument("validation", help="validation data file name")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("-s", "--step", type=int, default=10000, help="training steps")

    args = parser.parse_args()
    training_file = args.training
    validation_file = args.validation
    batch_size = args.batch_size
    steps = args.step

    # parameters
    vocab_size = 10000
    context_length = 512
    num_layers = 12
    num_heads = 8
    d_model = 512
    d_ff = int(8 / 3 * d_model)
    theta = 10000

    weight_decay = 0.05
    alpha_max = 6e-4
    alpha_min = 6e-5
    T_w = steps // 10
    T_c = steps

    max_grad_norm = 1.0

    # progress bar
    pbar = tqdm(range(1, steps + 1), desc="🚀 Training Model")

    # get device
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_id = 0
    device = f"cuda:{gpu_id}" if device_type == "cuda" else "cpu"

    # if use nvidia gpu, turn on the underlying TF32 accelerate，
    # it does not save VRAM, but accelerates computation
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # load training data
    training_data = np.load(file=training_file, mmap_mode="r")
    # load validation data
    validation_data = np.load(file=validation_file, mmap_mode="r")

    # initialize the model
    model = Transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=d_ff,
        rope_theta=theta,
        device=device,
    )
    model.to(device)

    # initialize the optimizer
    optimizer = AdamW(params=model.parameters(), lr=0.0, weight_decay=weight_decay)

    # main loop
    for t in pbar:
        # clear the grad
        optimizer.zero_grad()

        # get data using data loader
        inputs, targets = data_loader(
            dataset=training_data, batch_size=batch_size, context_length=context_length, device=device
        )

        # use mixed-precision-training to save VRAM, and accelerate computation
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            # forward pass
            logits = model(inputs)

            # calculate loss
            flatten_logits = rearrange(
                logits, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size"
            )
            flatten_targets = rearrange(targets, "batch_size context_length -> (batch_size context_length)")
            loss = cross_entropy(input=flatten_logits, targets=flatten_targets)

        # back prop
        loss.backward()

        # gradient clipping
        gradient_clipping(params=model.parameters(), max_norm=max_grad_norm)

        # calculate lr and update it to the optimizer
        lr = learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # update parameters
        optimizer.step()

        # update the prefix info of the progress bar
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{lr:.6e}"})

        # print training log
        if t % 100 == 0:
            print(f"Step: {t}/{steps}, LR: {lr:.6f}, Train loss: {loss.item():.4f}")

        # run on validation dataset
        if t % 1000 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                # sample 20 batches to avarage the val loss
                for _ in range(20):
                    val_inputs, val_targets = data_loader(
                        dataset=validation_data, batch_size=batch_size, context_length=context_length, device=device
                    )
                    # also use mixed-precision-training in validation dataset
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits = model(val_inputs)

                        flatten_logits = rearrange(
                            logits, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size"
                        )
                        flatten_targets = rearrange(
                            val_targets, "batch_size context_length -> (batch_size context_length)"
                        )
                        val_loss += cross_entropy(input=flatten_logits, targets=flatten_targets).item()

                val_loss /= 20

                # 3. use tqdm.write to replace print，preventing disrupting the progress bar display
                tqdm.write(f"🌟 Step: {t} | Validation loss: {val_loss:.4f}")

                # print(f"{'=' * 10}Step: {t}, Validation loss: {val_loss:.4f}{'=' * 10}")
            model.train()
        # save checkpoint during training
        if t == steps // 3 or t == steps * 2 // 3:
            checkpoint_path = f"./checkpoint/ckpt_step_{t}.pt"
            save_checkpoint(model=model, optimizer=optimizer, iteration=t, out=checkpoint_path)
            tqdm.write(f"💾 Checkpoint saved: {checkpoint_path}")

    # save final checkpoint
    checkpoint_path = "./checkpoint/ckpt_final.pt"
    save_checkpoint(model=model, optimizer=optimizer, iteration=t, out=checkpoint_path)
    print("🎉 Training Complete!")
