import unittest
from species_group import SpeciesGroup
    ### alt

class SpeciesGroupTests(unittest.TestCase):
    def test1(self):
        dic = {
            1: [2, 5],
            2: [1, 5, 6],
            3: [7, 8],
            4: [9, 10],
            5: [10, 1, 2],
            6: [2],
            7: [3],
            8: [3],
            9: [4],
            10: [5, 4]
        }

        # g_a = [1,2,3,4, 2,1,3,5,6, 3,1,2,5, 4,1,5,6] = [1,1,1,1, 2,2,2, 3,3,3, 4,4, 5,5,5] = [1,2,3,5]
        # g_b = [2,1,3,5,6, 1,2,3,4, 3,1,2,5 5,2,3,4,6, 6,2,4,5] = [1,1,1 2,2,2,2,2, 3,3,3,3, 4,4,4, 5,5,5,5, 6,6,6] = [2,3,5]
        # g_c = [3,1,2,5, 1,2,3,4, 2,1,3,5,6, 5,2,3,4,6] = [1,1,1, 2,2,2,2, 3,3,3,3, 4,4, 5,5,5] = [1,2,3,5]
        # g_d = [4,1,5,6, 1,2,3,4, 5,2,3,4,6, 6,2,4,5] = [1,1, 2,2,2, 3,3, 4,4,4,4, 5,5,5, 6,6,6] = [2,4,5,6]
        # g_e = [5,2,3,4,6, 2,1,3,5,6, 3,1,2,5, 4,1,5,6, 6,2,4,5] = [1,1,1, 2,2,2,2, 3,3,3, 4,4,4, 5,5,5,5,5, 6,6,6,6] = [2,5,6]
        # g_f = [6,2,4,5, 2,1,3,5,6, 4,1,5,6, 5,2,3,4,6] = [1,1, 2,2,2, 3,3, 4,4,4, 5,5,5,5, 6,6,6,6] = [5,6]
        # g_g = [7]
        
        groups = SpeciesGroup().iter(dic)
        
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0], set([1, 2, 4, 5, 6, 9, 10]))
        self.assertEqual(groups[1], set([3, 7, 8]))

    def test_double_run(self):
        dic = {
            1: [2, 5],
            2: [1, 5, 6],
            3: [7, 8],
            4: [9, 10],
            5: [10, 1, 2],
            6: [2],
            7: [3],
            8: [3],
            9: [4],
            10: [5, 4]
        }

        groups = SpeciesGroup().iter(dic)
        groups = SpeciesGroup().iter(dic)
        
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0], set([1, 2, 4, 5, 6, 9, 10]))
        self.assertEqual(groups[1], set([3, 7, 8]))

if __name__ == '__main__':
    unittest.main()