from cs336_basics.tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm

tokenizer = Tokenizer.from_files(
    vocab_filepath="tiny_vocab.json", 
    merges_filepath="tiny_merges.txt", 
    special_tokens=["<|endoftext|>"])

input_file_path = "data/TinyStoriesV2-GPT4-valid.txt"
token_ids_list = []
with open(input_file_path, mode="r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
    f.seek(0)
    # process line by line
    for line in tqdm(f, total=total_lines, desc="Encoding Validation Data"):
        temp = tokenizer.encode(text=line)
        token_ids_list.extend(temp)
        
# token_ids_list = tokenizer.encode(text=text_data)
print("converting to numpy array and saving...")
arr = np.array(token_ids_list, dtype=np.uint16)

output_file_path = "tiny_val_data.npy"
np.save(output_file_path, arr)
print(f"dataset process finished, total {len(arr)} tokens")
