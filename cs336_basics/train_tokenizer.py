import os
from collections import Counter, defaultdict
from typing import BinaryIO

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries


def train_bpe(
    file_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # initialize the vocab(0-255 bytes + special_tokens)
    vocab = {i: bytes([i]) for i in range(256)}
    cur_length = len(vocab)
    for i in range(len(special_tokens)):
        cur_index = i + cur_length
        special_token_bytes = special_tokens[i].encode("utf-8")
        vocab[cur_index] = special_token_bytes
    cur_length = len(vocab)
    target_merge_times = vocab_size - cur_length

    with open(file=file_path, mode="r", encoding="utf-8") as f:
        text = f.read()

    # clean out special_tokens
    escaped = [re.escape(token) for token in special_tokens]
    split_pattern = "|".join(escaped)

    split_chunks = [piece for piece in re.split(pattern=split_pattern, string=text) if piece]

    # do pre-tokenization for all chunks
    pre_token_counter = Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in split_chunks:
        matches = re.finditer(pattern=PAT, string=chunk)
        for match in matches:
            pre_token_counter[match.group()] += 1

    # convert this str->int counter to a index:int->dict dict
    word_dict = {}
    for word_id, (pre_token, freq) in enumerate(pre_token_counter.items()):
        pre_bytes = pre_token.encode("utf-8")
        bytes_tuple = tuple(bytes([i]) for i in pre_bytes)
        word_dict[word_id] = {"tokens": bytes_tuple, "freq": freq}

    # start merging
    pair2id = defaultdict(set)
    merges = []
    pair_counts = Counter()

    # initalize pair2id and pair_counts
    for id, token_dict in word_dict.items():
        token_tuple = token_dict["tokens"]
        freq = token_dict["freq"]

        for i in range(len(token_tuple) - 1):
            cur_pair = (token_tuple[i], token_tuple[i + 1])
            pair2id[cur_pair].add(id)
            pair_counts[cur_pair] += freq

    cur_merge_times = 0
    while cur_merge_times < target_merge_times:
        # find the best pair of this round
        best_pair_item = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        best_pair = best_pair_item[0]

        # add new merge to merges
        merges.append(best_pair)

        # add new_vocab to vocab dict
        new_vocab = best_pair[0] + best_pair[1]
        new_vocab_id = cur_length + cur_merge_times
        vocab[new_vocab_id] = new_vocab

        # get all the ids of pre_tokens where the best_pair exists, use deep-copy to prevent loop conflict
        bestpair_word_ids = pair2id[best_pair].copy()

        # merge through all pre_tokens where best_pair exists
        for word_id in bestpair_word_ids:
            old_tuple = word_dict[word_id]["tokens"]
            freq = word_dict[word_id]["freq"]

            # create the new tuple
            new_bytes_list = []
            i = 0
            while i <= (len(old_tuple) - 2):
                if (old_tuple[i], old_tuple[i + 1]) == best_pair:
                    new_bytes_list.append(old_tuple[i] + old_tuple[i + 1])
                    if i == len(old_tuple) - 3:
                        new_bytes_list.append(old_tuple[i + 2])
                    i += 2
                else:
                    new_bytes_list.append(old_tuple[i])
                    if i == len(old_tuple) - 2:
                        new_bytes_list.append(old_tuple[i + 1])
                    i += 1
            
            new_tuple = tuple(new_bytes_list)

            # update tokens tuple in word_dict
            word_dict[word_id]["tokens"] = new_tuple

            # update pair_counts
            old_tuple_pair_counts = Counter()
            new_tuple_pair_counts = Counter()
            for i in range(len(old_tuple) - 1):
                old_tuple_pair_counts[(old_tuple[i], old_tuple[i + 1])] += 1
            for i in range(len(new_tuple) - 1):
                new_tuple_pair_counts[(new_tuple[i], new_tuple[i + 1])] += 1
            new_tuple_pair_counts.subtract(old_tuple_pair_counts)
            for pair, count in new_tuple_pair_counts.items():
                pair_counts[pair] += freq * count
            # after all word_id's updating of pair_counts,
            # counts of best_pair will be reduced to 0,
            # counts of pair that does not overlap with best_pair will stay the same,
            # counts of new pair will be added to pair_counts

            # update pair-id mapping of new pairs to pair2id
            new_pairs = [pair for pair, value in new_tuple_pair_counts.items() if value > 0]
            for new_pair in new_pairs:
                pair2id[new_pair].add(word_id)
            
            # update pair-id mapping of destroyed pairs to pair2id
            destoryed_pairs = [pair for pair, value in new_tuple_pair_counts.items() if value < 0]
            for destoryed_pair in destoryed_pairs:
                pair2id[destoryed_pair].discard(word_id)


        # clear merged pair's data
        del pair2id[best_pair]
        del pair_counts[best_pair]

        # update cur_merge_times
        cur_merge_times += 1

    return vocab, merges

if __name__ == "__main__":
    pass
