from typing import Iterable, Iterator
from .train_tokenizer import load_tokenizer

class Tokenizer():
    def __init__(
            self, 
            vocab:dict[int, bytes], 
            merges:list[tuple[bytes,bytes]], 
            special_tokens:list[str]|None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # append special tokens to vocab if they are not there
        append_id = len(vocab)
        for special_token in special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in vocab.values():
                self.vocab[append_id] = special_token_bytes
                append_id += 1
        
        self.bytes2id = {token_bytes:id for id,token_bytes in self.vocab.items()}

    def from_files(
            cls,
            vocab_filepath:str,
            merges_filepath:str,
            special_tokens:list[str]|None = None):
        vocab, merges = load_tokenizer(vocab_filepath=vocab_filepath, merges_filepath=merges_filepath)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text:str)->list[int]:
        

    def encode_iterable(self, iterable:Iterable[str]) -> Iterator[int]:
        pass
    def decode(self, ids:list[int]) -> str:
        pass
