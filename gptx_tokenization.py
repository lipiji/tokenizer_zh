# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenization classes for GPTX."""

import sys
import os
import regex as re
import sentencepiece as spm
import jieba

class GPTXTokenizer(object):
    def __init__(self, model_file, max_len=None):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        
        self.encoder = {self.sp.id_to_piece(id):id for id in range(self.sp.get_piece_size())}
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.eod_id = self.encoder['<eod>'] 

    def __len__(self):
        return len(self.encoder)

    @property
    def vocab_size(self):
        return len(self.encoder)

    @property
    def vocab(self):
        return self.encoder

    @property
    def inv_vocab(self):
        return self.decoder

    @property
    def eod(self):
        return self.eod_id

    def tokenize(self, text, out_type=str):
        """ Tokenize a string. """
        seg_list = [w.replace(" ", "<space>").replace("\n", "<eol>") for w in jieba.cut(text, cut_all=False)]
        new_seg = " ".join(seg_list)
        return self.sp.encode(new_seg, out_type=out_type)

    def detokenize(self, tokens):
        text = self.sp.decode(tokens)
        text = text.replace(' ', '').replace('<space>', ' ').replace('<eol>', '\n')
        return text

    def encode(self, text):
        return self.tokenize(text, out_type=int) 

    def decode(self, tokens):
        return self.detokenize(tokens)
