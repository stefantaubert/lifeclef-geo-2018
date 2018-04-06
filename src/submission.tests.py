import unittest
import pandas as pd
import evaluation
import submission_maker
import numpy as np

class TestSubmissionEvaluationMethod(unittest.TestCase):
    def test1(self):
        classes = ["9", "3", "7"]

        prediction = [
            [0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5]
        ]
 
        submission = submission_maker.make_submission_array(classes, prediction)
 
        self.assertEqual(3*2, len(submission)) # Anzahl Klassen * Anzahl an Predictions (Größe des Validierungssets)

        ### glc_id,species_glc_id,probability,rank ### 
        self.assertEqual([1, "9", 0.5, 3], submission[0])
        self.assertEqual([1, "3", 0.6, 2], submission[1])
        self.assertEqual([1, "7", 0.7, 1], submission[2])
        self.assertEqual([2, "9", 0.7, 1], submission[3])
        self.assertEqual([2, "3", 0.6, 2], submission[4])
        self.assertEqual([2, "7", 0.5, 3], submission[5])
    
    def test_np_array(self):
        classes = np.array(["9", "3", "7"])

        prediction = [
            [0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5]
        ]
 
        submission = submission_maker.make_submission_array(classes, prediction)
 
        self.assertEqual(3*2, len(submission)) # Anzahl Klassen * Anzahl an Predictions (Größe des Validierungssets)

        ### glc_id,species_glc_id,probability,rank ### 
        self.assertEqual([1, "9", 0.5, 3], submission[0])
        self.assertEqual([1, "3", 0.6, 2], submission[1])
        self.assertEqual([1, "7", 0.7, 1], submission[2])
        self.assertEqual([2, "9", 0.7, 1], submission[3])
        self.assertEqual([2, "3", 0.6, 2], submission[4])
        self.assertEqual([2, "7", 0.5, 3], submission[5])

    def test_df(self):
        classes = ["9", "3", "7"]

        prediction = [
            [0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5]
        ]
 
        submission_df = submission_maker.make_submission_df(classes, prediction)
        self.assertEqual(3*2, len(submission_df.index))

        submission_matrix = submission_df.as_matrix()
        self.assertEqual([1, "9", 0.5, 3], list(submission_matrix[0]))
        self.assertEqual([1, "3", 0.6, 2], list(submission_matrix[1]))
        self.assertEqual([1, "7", 0.7, 1], list(submission_matrix[2]))
        self.assertEqual([2, "9", 0.7, 1], list(submission_matrix[3]))
        self.assertEqual([2, "3", 0.6, 2], list(submission_matrix[4]))
        self.assertEqual([2, "7", 0.5, 3], list(submission_matrix[5]))


if __name__ == '__main__':
    unittest.main()