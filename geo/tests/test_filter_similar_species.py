import unittest
import numpy as np

from geo.preprocessing.groups.filter_similar_species import filter_similar_species

class TestGetGroupsMethod(unittest.TestCase):
    similar_species = {
            # connected nodes 1
            1: [2,3],
            2: [1,3,4],
            3: [1,2,4,5],
            4: [2,3],
            5: [3,9],
            9: [5],
            # connected nodes 2
            6: [7],
            7: [6,8],
            8: [7],
        }

    def test_k1(self):
        groups = filter_similar_species(self.similar_species, k=1)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0], set([1,2,3,4,5,9]))
        self.assertEqual(groups[1], set([8,7,6]))

    def test_k2(self):
        groups = filter_similar_species(self.similar_species, k=2)

        self.assertEqual(len(groups), 6)
        self.assertEqual(groups[0], set([1,2,3,4]))
        self.assertEqual(groups[1], set([5]))
        self.assertEqual(groups[2], set([9]))
        self.assertEqual(groups[3], set([6]))
        self.assertEqual(groups[4], set([7]))
        self.assertEqual(groups[5], set([8]))

    def test_k3(self):
        groups = filter_similar_species(self.similar_species, k=3)

        self.assertEqual(len(groups), 9)
        self.assertEqual(groups[0], set([1]))
        self.assertEqual(groups[1], set([2]))
        self.assertEqual(groups[2], set([3]))
        self.assertEqual(groups[3], set([4]))
        self.assertEqual(groups[4], set([5]))
        self.assertEqual(groups[5], set([9]))
        self.assertEqual(groups[6], set([6]))
        self.assertEqual(groups[7], set([7]))
        self.assertEqual(groups[8], set([8]))

if __name__ == '__main__':
    unittest.main()