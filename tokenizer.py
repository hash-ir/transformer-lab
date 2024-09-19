"""Byte-pair tokenizer, used by OpenAI's GPT3.5 and GPT4"""

from collections import OrderedDict


def get_byte_pair_counts(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i] == pair[0] and ids[i + 1] == pair[1]):
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class Tokenizer:
    def __init__(self, num_merges=20, verbose=False):
        self.num_merges = num_merges
        self.merges = {}
        self.verbose = verbose

    def encode(self, text):
        tokens = text.encode("utf-8")  # raw bytes
        tokens = list(map(int, tokens))  # convert to list of integers
        ids = list(tokens)  # create a copy of tokens
        self.merges = OrderedDict()  # store a mapping of merges (int, int) -> int
        for i in range(self.num_merges):
            counts = get_byte_pair_counts(ids)
            pair = max(counts, key=counts.get)
            idx = 256 + i  # create new token for every merge step
            if self.verbose:
                print(f"Merging {pair} into a new token {idx}")
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx

        return ids

    def decode(self, ids):
        vocab = {idx: bytes([idx]) for idx in range(256)}  # original vocabulary
        # The iteration of items should be in the order of
        # their insertion. This is the default behavior in Python 3
        # but we use an OrderedDict explicitly here
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        tokens = b"".join(vocab[idx] for idx in ids)
        # handle UnicodeDecodeError by replacing the invalid
        # start byte to conform to utf-8 format
        text = tokens.decode("utf-8", errors="replace")
        return text


if __name__ == "__main__":
    text = """Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes 
    fear and awe into the hearts of programmers worldwide. We all know 
    we ought to “support Unicode” in our software (whatever that 
    means—like using wchar_t for all the strings, right?). But Unicode
    can be abstruse, and diving into the thousand-page Unicode Standard
    plus its dozens of supplementary annexes, reports, and notes can be
    more than a little intimidating. I don’t blame programmers for 
    still finding the whole thing mysterious, even 30 years after 
    Unicode’s inception.

    A few months ago, I got interested in Unicode and decided to spend
    some time learning more about it in detail. In this article, I’ll 
    give an introduction to it from a programmer’s point of view.

    I’m going to focus on the character set and what’s involved in
    working with strings and files of Unicode text. However, in this
    article I’m not going to talk about fonts, text 
    layout/shaping/rendering, or localization in detail—those are 
    separate issues, beyond my scope (and knowledge) here."""

    t = Tokenizer(verbose=True)
    tokens = t.encode(text)
    text2 = t.decode(tokens)
    assert text2 == text
