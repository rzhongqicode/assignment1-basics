import os
from typing import BinaryIO
import regex as re 
from collections import Counter, defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries


class Vocab():
    def __init__(self,  special_token):
        self.bytes_to_id = {bytes([i]):i for i in range(256)}
        self.bytes_to_id[special_token] = 256

def train(
        file_path: str,
        ):
    with open(file=file_path, mode="rb") as f:
        pass


def train_no_parallel(
        file_path:str,
        vocab_size:int,
        special_tokens:list[str]
):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(file=file_path, mode="rb") as f:
        if not f:
            Exception("No such file")
        else:
            text = f.read()
            #clean out special_tokens
            escaped = [re.escape(token) for token in special_tokens]
            split_pattern = "|".join(escaped)
            
            split_chunks = [piece for piece in re.split(pattern=split_pattern, string=text) if piece]

            pre_token_counter = Counter()
            #do pre-tokenization for all chunks
            for chunk in split_chunks:
                matches = re.finditer(pattern=PAT, string=chunk)
                for match in matches:
                    pre_token_counter[match.group()] += 1
            
            #convert this str->int counter to a index:int->dict dict
            word_dict = {}
            for word_id, (pre_token, freq) in enumerate(pre_token_counter.items()):
                pre_bytes = pre_token.encode("utf-8")
                bytes_tuple = tuple(bytes([i]) for i in pre_bytes)
                word_dict[word_id] = {
                    "tokens":bytes_tuple,
                    "freq":freq
                }

            # split_word_dict = {}
            # for pre_token, freq in pre_token_counter.items():
            #     pre_bytes = pre_token.encode("utf-8")

            #     bytes_tuple = tuple(bytes([i]) for i in pre_bytes)
            #     split_word_dict[bytes_tuple] = freq
            
            # pair_counts = Counter()
            # merges = []

            # for cur_tuple, freq in split_word_dict.items():
            #     for i in range(len(cur_tuple)-1):
            #         pair_counts[(cur_tuple[i],cur_tuple[i+1])] += freq
            # best_pair_item = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
            # best_pair = best_pair_item[0]
            # new_merge = best_pair[0]+best_pair[1]
            # merges.append(new_merge)
            pair2id=defaultdict(set)
            merges=[]
            pair_counts = Counter()
            
            for id, token_dict in word_dict.items():
                token_tuple = token_dict["tokens"]
                freq = token_dict["freq"]

                for i in range(len(token_tuple) - 1):
                    cur_pair = (token_tuple[i], token_tuple[i+1])
                    pair2id[cur_pair].add(id)
                    pair_counts[cur_pair] += freq

            best_pair_item = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
            best_pair = best_pair_item[0]
            merges.append(best_pair)
            

        


            



if __name__ == "__main__":
    train(file_path="data/TinyStoriesV2-GPT4-train.txt")
    