class BasicTokenizer():
    '''
    Basic tokenizer that treats text as a single large string. \n
    Best compression, but creates unstructured merges 
    that do not account for grammatical rules (e.g. punctuation).
    '''

    def __init__(self):
        self.vocab = {}  # dict mapping bigrams to respective NEW tokens (>255)
        self.byte_vocab = {}  # dict mapping ALL tokens to respective binary representation

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        '''
        Identifies the most common byte-bigrams, \n
        creates new tokens for these bigrams and \n
        stores the mappings in `self.vocab`.
        '''
        assert vocab_size >= 256
        self.vocab = {}

        ids = list(text.encode('utf-8'))  # list of int repr of bytes
        idx = 256
        while idx < vocab_size:
            stats = self._get_stats(ids)
            pair = list(stats.keys())[0]
            ids = self._merge(ids, pair, idx)
            self.vocab[pair] = idx
            idx += 1

        if verbose:
            for (p0, p1), idx in self.vocab.items():
                print(
                    f"{self.decode([p0])}|{self.decode([p1])} \t {(p0,p1)} --> {idx}")

    def encode(self, text: str) -> list:
        '''
        Encodes text to bytes using "utf-8"-encoding, \n
        merges byte-bigrams according to trained `self.vocab` and \n
        returns list of tokens (int).
        '''
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = list(stats.keys())[0]
            if pair not in self.vocab:
                break
            idx = self.vocab[pair]
            ids = self._merge(ids, pair, idx)
        return ids

    def decode(self, ids: list) -> str:
        '''
        Substitutes tokens (int) with their byte-representation 
        given by `self.byte_vocab`, \n
        decodes byte-string using "utf-8"-encoding and \n
        returns decoded text.
        '''
        self._get_byte_vocab()
        tokens = b''.join(self.byte_vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text

    # def _get_stats(self, ids):
    #     stats = {i:0 for i in set(zip(ids, ids[1:]))}
    #     for i in zip(ids, ids[1:]):
    #         stats[i] += 1
    #     sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
    #     return sorted_stats

    def _get_stats(self, idslist: list) -> dict:
        '''
        Counts occurrences of bigrams in words of `idslist`.
        Words are elements from splitting the text with regex 
        (or the entire text if regex is not used).
        '''
        if type(idslist[0]) != list:
            # if regex was not used, ids becomes a word of idslist
            idslist = [idslist]
        stats = {}
        for w in idslist:
            if len(w) > 1:
                for pair in zip(w, w[1:]):
                    stats[pair] = stats.get(pair, 0) + 1
        sorted_stats = dict(
            sorted(stats.items(), key=lambda x: x[1], reverse=True))
        return sorted_stats

    def _merge(self, ids: list, pair: tuple, idx: int) -> list:
        '''
        Replaces all occurrences of `pair` in `ids` with `idx`. 
        Returns a new merged list.
        '''
        merged_ids = []
        i = 0
        while i < len(ids)-1:
            if (ids[i], ids[i+1]) == pair:
                merged_ids += [idx]
                i += 2
            else:
                merged_ids += [ids[i]]
                i += 1
        if i == len(ids) - 1:
            merged_ids += [ids[i]]
        return merged_ids

    def _get_byte_vocab(self):
        '''
        Creates a mapping from ALL tokens to their respective binary representation.
        Saves it in `self.byte_vocab`.
        '''
        self.byte_vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.vocab.items():
            self.byte_vocab[idx] = self.byte_vocab[p0] + self.byte_vocab[p1]
