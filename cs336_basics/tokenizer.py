from typing import Iterable, Iterator
from .train_tokenizer import load_tokenizer
import regex as re

class Tokenizer():
    def __init__(
            self, 
            vocab:dict[int, bytes], 
            merges:list[tuple[bytes,bytes]], 
            special_tokens:list[str]|None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # append special tokens to vocab if they are not there
        append_id = len(vocab)
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in vocab.values():
                self.vocab[append_id] = special_token_bytes
                append_id += 1
        
        self.bytes2id = {token_bytes:id for id,token_bytes in self.vocab.items()}

    @classmethod
    def from_files(
            cls,
            vocab_filepath:str,
            merges_filepath:str,
            special_tokens:list[str]|None = None):
        vocab, merges = load_tokenizer(vocab_filepath=vocab_filepath, merges_filepath=merges_filepath)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text:str)->list[int]:
        if self.special_tokens:
            # split text using special tokens
            escaped = [re.escape(token) for token in self.special_tokens]
            split_pattern = "|".join(escaped)
            split_pattern = f"({split_pattern})"
            split_chunks = [piece for piece in re.split(pattern=split_pattern, string=text) if piece]
        else:
            split_chunks = [text]

        # create a ranking list of merges
        merge_rank = {merge_pair:index for index, merge_pair in enumerate(self.merges)}

        # do pre-tokenization for non-special_token chunks
        id_list = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in split_chunks:
            if chunk in self.special_tokens:
                chunk_bytes = chunk.encode("utf-8")
                id = self.bytes2id[chunk_bytes]
                id_list.append(id)
            else:
                matches = re.finditer(pattern=PAT, string=chunk)
                for match in matches:
                    pre_token = match.group()
                    pre_token_bytes = pre_token.encode("utf-8")
                    bytes_tuple = tuple(bytes([i]) for i in pre_token_bytes)
                    while True:
                        cur_rank_dict = {}
                        for i in range(len(bytes_tuple)-1):
                            cur_pair = (bytes_tuple[i],bytes_tuple[i+1])
                            if cur_pair in merge_rank:
                                cur_rank_dict[cur_pair] = merge_rank[cur_pair]
                        if cur_rank_dict:
                            merge_pair = min(cur_rank_dict.items(), key=lambda x: x[1])[0]
                            # create the new tuple
                            new_bytes_list = []
                            i = 0
                            while i < len(bytes_tuple):
                                if i < len(bytes_tuple) - 1 and (bytes_tuple[i], bytes_tuple[i + 1]) == merge_pair:
                                    new_bytes_list.append(bytes_tuple[i] + bytes_tuple[i + 1])
                                    i += 2
                                else:
                                    new_bytes_list.append(bytes_tuple[i])
                                    i += 1
                            # update bytes_tuple
                            bytes_tuple = tuple(new_bytes_list)
                        else:
                            for token in bytes_tuple:
                                id_list.append(self.bytes2id[token])
                            break
        return id_list
                

    def encode_iterable(self, iterable:Iterable[str]) -> Iterator[int]:
        pass
    def decode(self, ids:list[int]) -> str:
        pass
