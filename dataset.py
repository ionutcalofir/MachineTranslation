import os
import re
import unicodedata
import gensim.downloader
import torch.utils.data
import numpy as np
from nltk.stem.porter import *
from absl import logging
from sklearn.model_selection import train_test_split
from tokenizers import BertWordPieceTokenizer

MAX_TOKENS = 50
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

class MyCollator:
    def __call__(self, batch):
        max_sent_l0 = max(len(sent[0]) for sent in batch)
        max_sent_l1 = max(len(sent[1]) for sent in batch)

        src = []
        trg = []
        for sent in batch:
            sent_l0 = sent[0]
            sent_l1 = sent[1]

            sent_l0.insert(0, SOS_IDX)
            sent_l1.insert(0, SOS_IDX)

            sent_l0.append(EOS_IDX)
            sent_l1.append(EOS_IDX)

            sent_l0.extend((max_sent_l0 - (len(sent_l0) - 2)) * [PAD_IDX])
            sent_l1.extend((max_sent_l1 - (len(sent_l1) - 2)) * [PAD_IDX])

            src.append(sent_l0)
            trg.append(sent_l1)

        src = np.array(src)
        trg = np.array(trg)

        return torch.from_numpy(src), torch.from_numpy(trg)

class Processor:
    def __init__(self, text_preprocessor):
        self._text_preprocessor = text_preprocessor
        if text_preprocessor == 'stemming':
            self._stemmer = PorterStemmer()
        elif text_preprocessor == 'bert_tokenizer':
            self._tokenizer_l0 = BertWordPieceTokenizer(vocab_file='./tokenizer/vocab_files/bert-wordpiece-en-vocab.txt',
                clean_text=True, strip_accents=True, lowercase=True,)
            self._tokenizer_l1 = BertWordPieceTokenizer(vocab_file='./tokenizer/vocab_files/bert-wordpiece-ro-vocab.txt',
                clean_text=True, strip_accents=True, lowercase=True,)

    def unicodeToAscii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def tokenize(self, sent, lang):
        if self._text_preprocessor == 'bert_tokenizer':
            if lang == 0:
                output = self._tokenizer_l0.encode(sent)
            else:
                output = self._tokenizer_l1.encode(sent)
            tokens = output.tokens[1:-1]
            sent = []
            for token in tokens:
                if token == "[UNK]" or token == "[MASK]":
                    token = '<unk>'
                elif token == "[PAD]":
                    token = '<pad>'

                sent.append(token)
        else:
            sent = self.unicodeToAscii(sent.lower().strip())

            sent = re.sub(r"([.!?])", r" \1", sent)
            sent = re.sub(r"[^a-zA-Z.!?]+", r" ", sent)

            sent = [word for word in sent.split()]
            if self._text_preprocessor == 'stemming' and lang == 0: # we don't apply stemming on l1
                sent = [self._stemmer.stem(word) for word in sent]

        return sent

    def tokens2ids(self, tokens, vocab):
        tokens = [vocab.get(token, vocab['<unk>'])[1] for token in tokens[:MAX_TOKENS]]

        return tokens

class BaseDataset():
    def __init__(self, data_path, name_suffix, use_w2v, text_preprocessor):
        self._data_path = data_path
        self._name_suffix = name_suffix
        self._processor = Processor(text_preprocessor)
        self._use_w2v = use_w2v

        self.train = [(line.strip().split('\t')[0], line.strip().split('\t')[1])
                        for line in open(os.path.join(data_path, 'train_{}.txt'.format(self._name_suffix)), 'r')]
        self.val = [(line.strip().split('\t')[0], line.strip().split('\t')[1])
                        for line in open(os.path.join(data_path, 'val_{}.txt'.format(self._name_suffix)), 'r')]
        self.test = [(line.strip().split('\t')[0], line.strip().split('\t')[1])
                        for line in open(os.path.join(data_path, 'test_{}.txt'.format(self._name_suffix)), 'r')]

        self.vocab_l0 = self._build_vocab(0)
        self.vocab_l1 = self._build_vocab(1)

        if self._use_w2v:
            self.w2v_l0 = self._build_word2vec(self.vocab_l0)

    def _build_word2vec(self, vocab):
        logging.info('Loading w2v')
        w2v = gensim.downloader.load('word2vec-google-news-300')
        logging.info('Done!')

        embds = np.random.normal(size=(len(vocab), 300))
        for key, val in vocab.items():
            pos = val[1]
            if key in w2v:
                embds[pos, :] = w2v[key][:]

        return embds

    def _build_vocab(self, l):
        vocab = {}
        preprocess_vocab = {'<eos>': (0, EOS_IDX),
                            '<pad>': (0, PAD_IDX),
                            '<sos>': (0, SOS_IDX),
                            '<unk>': (0, UNK_IDX)}
        if self._processor._text_preprocessor == 'bert_tokenizer':
            assert self._processor._tokenizer_l0.token_to_id("[CLS]") == self._processor._tokenizer_l1.token_to_id("[CLS]")
            assert self._processor._tokenizer_l0.token_to_id("[SEP]") == self._processor._tokenizer_l1.token_to_id("[SEP]")
            assert self._processor._tokenizer_l0.token_to_id("[PAD]") == self._processor._tokenizer_l1.token_to_id("[PAD]")
            assert self._processor._tokenizer_l0.token_to_id("[UNK]") == self._processor._tokenizer_l1.token_to_id("[UNK]")
            global UNK_IDX
            global PAD_IDX
            global SOS_IDX
            global EOS_IDX

            preprocess_vocab = {'<eos>': (0, self._processor._tokenizer_l0.token_to_id("[SEP]")),
                                '<pad>': (0, self._processor._tokenizer_l0.token_to_id("[PAD]")),
                                '<sos>': (0, self._processor._tokenizer_l0.token_to_id("[CLS]")),
                                '<unk>': (0, self._processor._tokenizer_l0.token_to_id("[UNK]"))}

            UNK_IDX = preprocess_vocab['<unk>'][1]
            PAD_IDX = preprocess_vocab['<pad>'][1]
            SOS_IDX = preprocess_vocab['<sos>'][1]
            EOS_IDX = preprocess_vocab['<eos>'][1]

        avg_sent = 0
        for sent in self.train:
            tokens = self._processor.tokenize(sent[l], lang=l)
            avg_sent += len(tokens)
            for token in tokens:
                if token in ['<pad>', '<unk>']:
                    continue

                nr_token, pos = vocab.get(token, (0, len(vocab)))
                nr_token += 1
                vocab[token] = (nr_token, pos)

        keys = sorted(vocab.keys())
        for key in keys:
            if vocab[key][0] >= 3:
                preprocess_vocab[key] = (vocab[key][0], len(preprocess_vocab))
        logging.info('Avg tokens per sent for l{}: {}'.format(l, avg_sent / len(self.train)))

        return preprocess_vocab

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, vocab_l0, vocab_l1, phase, text_preprocessor):
        self._data = data
        self._vocab_l0 = vocab_l0
        self._vocab_l1 = vocab_l1
        self._phase = phase

        self._processor = Processor(text_preprocessor)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sent0 = self._data[idx][0]
        sent1 = self._data[idx][1]

        tokens0 = self._processor.tokenize(sent0, lang=0)
        tokens1 = self._processor.tokenize(sent1, lang=1)

        tokens0_to_ids = self._processor.tokens2ids(tokens0, self._vocab_l0)
        tokens1_to_ids = self._processor.tokens2ids(tokens1, self._vocab_l1)

        return tokens0_to_ids, tokens1_to_ids
