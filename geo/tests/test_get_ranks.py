import unittest
import pandas as pd
import get_ranks

class TestGetRanksMethod(unittest.TestCase):
    '''Testcases for the get_ranks()- and get_ranks_df()-method.'''
    def test_basic(self):
        submissions = [
            [1, 5, 0.9, 1],
            [1, 4, 0.9, 2],
            [1, 6, 0.5, 3],
            [2, 6, 0.8, 1],
            [2, 5, 0.7, 2],
            [2, 4, 0.7, 3],
            [3, 6, 0.8, 1],
            [3, 5, 0.7, 2],
            [3, 4, 0.7, 3],
            [4, 6, 0.8, 1],
            [4, 5, 0.7, 2],
            [4, 4, 0.7, 3],
        ]

        solution = [
            4,
            6,
            3,
            4,
        ]

        top_n = 3

        ranks = get_ranks.get_ranks(submissions, solution, top_n)

        self.assertEqual(4, len(ranks))
        self.assertEqual(2, ranks[0])
        self.assertEqual(1, ranks[1])
        self.assertEqual(0, ranks[2])
        self.assertEqual(3, ranks[3])

    def test_df(self):
        submissions = [
            [1, 5, 0.9, 1],
            [1, 4, 0.9, 2],
            [1, 6, 0.5, 3],
            [2, 6, 0.8, 1],
            [2, 5, 0.7, 2],
            [2, 4, 0.7, 3],
            [3, 6, 0.8, 1],
            [3, 5, 0.7, 2],
            [3, 4, 0.7, 3],
            [4, 6, 0.8, 1],
            [4, 5, 0.7, 2],
            [4, 4, 0.7, 3],
        ]

        solution = [
            4,
            6,
            3,
            4,
        ]

        top_n = 3

        submission_df = pd.DataFrame(submissions, columns = ['patch_id', 'species_glc_id', 'probability', 'rank'])
        ranks = get_ranks.get_ranks_df(submission_df, solution, top_n)

        self.assertEqual(4, len(ranks))
        self.assertEqual(2, ranks[0])
        self.assertEqual(1, ranks[1])
        self.assertEqual(0, ranks[2])
        self.assertEqual(3, ranks[3])

if __name__ == '__main__':
    unittest.main()