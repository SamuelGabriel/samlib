import unittest

import random
from copy import copy

from samlib import utils



class TestUtils(unittest.TestCase):
    def test_shuffle_list_with_seed(self):
        l = list(range(100))
        l_copy = copy(l)
        s = random.getstate()
        utils.shufflelist_with_seed(l, seed='123')

        self.assertEqual(s, random.getstate())
        self.assertNotEqual(l,l_copy)

        utils.shufflelist_with_seed(l_copy, seed='123')
        self.assertEqual(l,l_copy)

    def test_chunker(self):
        l = list(range(10))
        self.assertEqual(list(utils.chunker(3,l)),[[0,1,2],[3,4,5],[6,7,8],[9]])
        l = range(10)
        self.assertEqual(list(utils.chunker(3,l)),[[0,1,2],[3,4,5],[6,7,8],[9]])
        l = range(9)
        self.assertEqual(list(utils.chunker(3,l)),[[0,1,2],[3,4,5],[6,7,8]])



if __name__ == '__main__':
    unittest.main()
