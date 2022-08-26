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


def set_locals_in_self(locals):
    self = locals['self']
    for var_name, val in locals.items():
        if var_name != 'self': setattr(self, var_name, val)