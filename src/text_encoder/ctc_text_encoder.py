import re
from string import ascii_lowercase

from pyctcdecode import Alphabet, BeamSearchDecoderCTC
from typing import List, NamedTuple
import torch
import math
import numpy as np
from collections import defaultdict

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier
# DONE

class Hypo(NamedTuple):
    text: str
    prob: float

class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, _default_beam_size=100, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.decoder = BeamSearchDecoderCTC(Alphabet(self.vocab, False), None)
        self.default_beam_size = _default_beam_size

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        result_string = ''
        prev = self.EMPTY_TOK
        for ind in inds:
            char = self.ind2char[ind]
            if char == self.EMPTY_TOK:
                prev = char
                continue

            if (prev == self.EMPTY_TOK or char != result_string[-1]):
                result_string += char
                prev = char

        return result_string

    def make_one_step(self, dp, next_token_probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if last_char == cur_char or cur_char == self.EMPTY_TOK:
                    new_prefix = prefix
                elif cur_char != self.EMPTY_TOK:
                    new_prefix = prefix + cur_char
                new_dp[(new_prefix, cur_char)] += v * next_token_prob
        return new_dp

    def ctc_beam_search_for_test(self, inds: torch.Tensor) -> str:
        dp = {
            ("", self.EMPTY_TOK): 1.0,
        }
        for prob in inds:
            dp = self.make_one_step(dp, prob)
            dp = dict(sorted(list(dp.items(), key=lambda x: -x[1]))[:self.default_beam_size])

        dp = [
            Hypo(prefix, proba)
            for (prefix, _), proba in sorted(dp.items(), key=lambda x: -x[1])
        ]
        return dp

    def ctc_beam_search(self, inds: torch.Tensor) -> str:
        return self.decoder.decode(inds.cpu().numpy(), beam_width=self.default_beam_size)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text