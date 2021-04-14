import random
import itertools


def shufflelist_with_seed(lis, seed='2020'):
    s = random.getstate()
    random.seed(seed)
    random.shuffle(lis)
    random.setstate(s)


def chunker(chunk_length, iterable):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, chunk_length))
        if not chunk:
            return
        yield chunk