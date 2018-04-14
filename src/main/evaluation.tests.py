import unittest
import pandas as pd
import evaluation
import submission_maker
import numpy as np
import mrr
import evaluation

class TestEvaluationRanksMethod(unittest.TestCase):
    def test1(self):
        classes = ["9", "3", "7"]

        prediction = [
            [0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5]
        ]
 
        submission_df = submission_maker.make_submission_df(classes, prediction)

        solution = [
            "3",
            "9"
        ]

        ranks = evaluation.get_ranks(submission_df, solution, len(classes))

        self.assertEqual(2, len(ranks))
        self.assertEqual(2, ranks[0]) # 3 -> 0.6 hat Rang 2
        self.assertEqual(1, ranks[1]) # 9 -> 0.7 hat Rang 1

        print(mrr.mrr_score(ranks))

    def test2(self):
        classes = [1, 2, 3, 4]

        prediction = [
            [1, 0.9, 0.8, 0.7],
            [0.9, 0.8, 1, 0.7], 
            [0.9, 1, 0.8, 0.7], 
        ]
 
        submission_df = submission_maker.make_submission_df(classes, prediction)

        solution = [
            1,
            3,
            1
        ]

        ranks = evaluation.get_ranks(submission_df, solution, len(classes))

        self.assertEqual(len(ranks), 3)
        self.assertEqual(ranks[0], 1) # Klasse 1 wurde mit 1% predicted und hat demnach Rang 1
        self.assertEqual(ranks[1], 1) # Klasse 3 wurde mit 100% predicted und hat demnach Rang 1
        self.assertEqual(ranks[2], 2) # Klasse 1 wurde mit 90% predicted und hat demnach Rang 2

        print(mrr.mrr_score(ranks))

    def test_np_array(self):
        classes = ["9", "3", "7"]

        prediction = [
            [0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5]
        ]
 
        submission_df = submission_maker.make_submission_df(classes, prediction)

        solution = np.array([
            "3",
            "9"
        ])

        ranks = evaluation.get_ranks(submission_df, solution, len(classes))

        self.assertEqual(2, len(ranks))
        self.assertEqual(2, ranks[0]) # 3 -> 0.6 hat Rang 2
        self.assertEqual(1, ranks[1]) # 9 -> 0.7 hat Rang 1

        print(mrr.mrr_score(ranks))


if __name__ == '__main__':
    unittest.main()