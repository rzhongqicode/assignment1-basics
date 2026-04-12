from cs336_basics.tokenizer import Tokenizer
import numpy as np

tokenizer = Tokenizer.from_files(
    vocab_filepath="tiny_vocab.json", 
    merges_filepath="tiny_merges.txt", 
    special_tokens=["<|endoftext|>"])

input_file_path = "data/TinyStoriesV2-GPT4-train.txt"
with open(input_file_path, mode="r", encoding="utf-8") as f:
    text_data = f.read()

token_ids_list = tokenizer.encode(text=text_data)

arr = np.array(token_ids_list, dtype=np.uint16)

output_file_path = "tiny_train_data.npy"
np.save(output_file_path, arr)
