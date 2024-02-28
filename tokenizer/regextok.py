from basic import BasicTokenizer
import regex as re


class RegexTokenizer(BasicTokenizer):
    '''
    Advanced tokenizer that introduces hard-coded structure into 
    the tokenization process. \n
    Structure is specified by `self.split_pattern` 
    (for more info: https://www.regular-expressions.info/refquick.html)
    '''

    def __init__(self):
        super().__init__()
        self.split_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        # Specifies hard-coded structure in text

    def train(self, text, vocab_size, verbose=False):
        '''
        Identifies the most common byte-bigrams (amoung all words!), \n
        creates new tokens for these bigrams and \n
        stores the mappings in `self.vocab`.
        '''
        assert vocab_size >= 256
        self.vocab = {}

        h = re.compile(self.split_pattern)
        words = re.findall(h, text)
        idslist = [list(w.encode('utf-8')) for w in words]

        idx = 256
        while idx < vocab_size:
            stats = self._get_stats(idslist)
            pair = list(stats.keys())[0]
            for iw in range(len(idslist)):
                word = idslist[iw]
                idslist[iw] = self._merge(word, pair, idx)
            self.vocab[pair] = idx
            idx += 1

        if verbose:
            for pair, idx in self.vocab.items():
                print(
                    f"{self.decode([pair[0]])}|{self.decode([pair[1]])} \t {pair} --> {idx}")

    def encode(self, text):
        h = re.compile(self.split_pattern)
        words = re.findall(h, text)
        idslist = [list(w.encode('utf-8')) for w in words]

        encids = []
        for word in idslist:
            while len(word) >= 2:
                stats = self._get_stats(word)
                merging_pairs = [i for i in stats.keys() if i in self.vocab]
                if len(merging_pairs) == 0:
                    break
                for pair in merging_pairs:
                    idx = self.vocab[pair]
                    word = self._merge(word, pair, idx)
            encids += word
        return encids
