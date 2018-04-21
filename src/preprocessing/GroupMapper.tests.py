import unittest
from GroupMapper import map_groups

class TestMapGroupsMethod(unittest.TestCase):
    def test1(self):
        original = [1,2,3,4]
        groups = {0:[1,2], 1:[3,4]}
        
        result = map_groups(original, groups)

        self.assertEqual(result, [0,0,1,1])

    def test2(self):
        original = [64,34,48,374,45,6]
        groups = {8:[64,48], 48:[34,374], 4:[45,6]}
        
        result = map_groups(original, groups)

        self.assertEqual(result, [8,48,8,48,4,4])

    def test_groups_have_more_species(self):
        original = [1,2,3]
        groups = {0:[1,2], 1:[3,4]}
        
        result = map_groups(original, groups)

        self.assertEqual(result, [0,0,1])

    def test_groups_have_less_species(self):
        original = [1,2,3,4,5]
        groups = {0:[1,2], 1:[3,4]}

        try:
            map_groups(original, groups)
        except AssertionError:
            pass
        else:
            self.fail()

if __name__ == '__main__':
    unittest.main()