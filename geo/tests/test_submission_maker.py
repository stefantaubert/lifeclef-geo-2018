import unittest
import pandas as pd
import submission_maker
import numpy as np

class TestMakeSubmissionMethods(unittest.TestCase):
    '''Contains tests for the methods: _make_submission_groups(), _make_submission() and make_submission_df() of the module submission_maker.'''
    def test_group_submission_top_3(self):
        top_n = 3
        groups = {11: [4,5], 22:[6]}
        props = {4: 0.4, 5: 0.5, 6: 0.6}
        groups_map = [11,22]
        prediction = [
            [0.9, 0.5],
            [0.7, 0.8],
        ]

        glc_ids = [1,2]

        submission = submission_maker._make_submission_groups(top_n, groups_map, prediction, glc_ids, groups, props)
        
        self.assertEqual(top_n*len(prediction), len(submission)) # top_n * Anzahl an Predictions (Größe des Validierungssets)

        ### glc_id,species_glc_id,probability,rank ### 
        self.assertEqual([1, 5, 0.9, 1], submission[0])
        self.assertEqual([1, 4, 0.9, 2], submission[1])
        self.assertEqual([1, 6, 0.5, 3], submission[2])
        self.assertEqual([2, 6, 0.8, 1], submission[3])
        self.assertEqual([2, 5, 0.7, 2], submission[4])
        self.assertEqual([2, 4, 0.7, 3], submission[5])

    def test_group_submission_top_2(self):
        top_n = 2
        groups = {11: [4,5], 22:[6]}
        props = {4: 0.4, 5: 0.5, 6: 0.6}
        groups_map = [11,22]
        prediction = [
            [0.9, 0.5],
            [0.7, 0.8],
        ]

        glc_ids = [1,2]

        submission = submission_maker._make_submission_groups(top_n, groups_map, prediction, glc_ids, groups, props)
        
        self.assertEqual(top_n*len(prediction), len(submission)) 

        ### glc_id,species_glc_id,probability,rank ### 
        self.assertEqual([1, 5, 0.9, 1], submission[0])
        self.assertEqual([1, 4, 0.9, 2], submission[1])
        self.assertEqual([2, 6, 0.8, 1], submission[2])
        self.assertEqual([2, 5, 0.7, 2], submission[3])

    def test_group_submission_top_1(self):
        top_n = 1
        groups = {11: [4,5], 22:[6]}
        props = {4: 0.4, 5: 0.5, 6: 0.6}
        groups_map = [11,22]
        prediction = [
            [0.9, 0.5],
            [0.7, 0.8],
        ]

        glc_ids = [1,2]

        submission = submission_maker._make_submission_groups(top_n, groups_map, prediction, glc_ids, groups, props)
        
        self.assertEqual(top_n*len(prediction), len(submission)) 

        ### glc_id,species_glc_id,probability,rank ### 
        self.assertEqual([1, 5, 0.9, 1], submission[0])
        self.assertEqual([2, 6, 0.8, 1], submission[1])

    def test_convert_to_int(self):
        top_n = 3
        classes = ["9", "3", "7"]

        prediction = [
            [0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5]
        ]

        glc_ids = [2.0,4.0]
 
        submission = submission_maker._make_submission(top_n, classes, prediction, glc_ids)
 
        self.assertEqual(3*2, len(submission)) # Anzahl Klassen * Anzahl an Predictions (Größe des Validierungssets)

        ### glc_id,species_glc_id,probability,rank ### 
        self.assertEqual([2, 7, 0.7, 1], submission[0])
        self.assertEqual([2, 3, 0.6, 2], submission[1])
        self.assertEqual([2, 9, 0.5, 3], submission[2])
        self.assertEqual([4, 9, 0.7, 1], submission[3])
        self.assertEqual([4, 3, 0.6, 2], submission[4])
        self.assertEqual([4, 7, 0.5, 3], submission[5])
    
    def test_top_3(self):
        species_map = [5, 6, 7]

        prediction = [
            [0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5],
            [0.6, 0.7, 0.5],
        ]

        glc_ids = [2,3,4]

        top_n = 3
 
        submission = submission_maker._make_submission(top_n, species_map, prediction, glc_ids)
 
        self.assertEqual(9, len(submission)) # top_n * Anzahl an Predictions (Größe des Validierungssets bzw len(glc_ids))

        ### glc_id,species_glc_id,probability,rank ### 
        self.assertEqual([2, 7, 0.7, 1], submission[0])
        self.assertEqual([2, 6, 0.6, 2], submission[1])
        self.assertEqual([2, 5, 0.5, 3], submission[2])
        self.assertEqual([3, 5, 0.7, 1], submission[3])
        self.assertEqual([3, 6, 0.6, 2], submission[4])
        self.assertEqual([3, 7, 0.5, 3], submission[5])
        self.assertEqual([4, 6, 0.7, 1], submission[6])
        self.assertEqual([4, 5, 0.6, 2], submission[7])
        self.assertEqual([4, 7, 0.5, 3], submission[8])

    def test_top_2(self):
        species_map = [5, 6, 7]

        prediction = [
            [0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5],
            [0.6, 0.7, 0.5],
        ]

        glc_ids = [2,3,4]

        top_n = 2
 
        submission = submission_maker._make_submission(top_n, species_map, prediction, glc_ids)
 
        self.assertEqual(6, len(submission)) # top_n * Anzahl an Predictions (Größe des Validierungssets bzw len(glc_ids))

        ### glc_id,species_glc_id,probability,rank ### 
        self.assertEqual([2, 7, 0.7, 1], submission[0])
        self.assertEqual([2, 6, 0.6, 2], submission[1])
        self.assertEqual([3, 5, 0.7, 1], submission[2])
        self.assertEqual([3, 6, 0.6, 2], submission[3])
        self.assertEqual([4, 6, 0.7, 1], submission[4])
        self.assertEqual([4, 5, 0.6, 2], submission[5])

    def test_top_1(self):
        species_map = [5, 6, 7]

        prediction = [
            [0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5],
            [0.6, 0.7, 0.5],
        ]

        glc_ids = [2,3,4]

        top_n = 1
 
        submission = submission_maker._make_submission(top_n, species_map, prediction, glc_ids)
 
        self.assertEqual(3, len(submission)) # top_n * Anzahl an Predictions (Größe des Validierungssets bzw len(glc_ids))

        ### glc_id,species_glc_id,probability,rank ### 
        self.assertEqual([2, 7, 0.7, 1], submission[0])
        self.assertEqual([3, 5, 0.7, 1], submission[1])
        self.assertEqual([4, 6, 0.7, 1], submission[2])

    def test2(self):
        top_n = 4
        classes = [5, 4, 9, 1]
        glc_ids = [1, 4, 3]

        prediction = [
            [1, 0, 0, 0],
            [0, 0, 1, 0], 
            [0, 1, 0, 0], 
        ]
 
        submission = submission_maker._make_submission(top_n, classes, prediction, glc_ids)
 
        self.assertEqual(4*3, len(submission)) # Anzahl Klassen * Anzahl an Predictions (Größe des Validierungssets)

        ### glc_id,species_glc_id,probability,rank ### 
        self.assertEqual([1, 5, 1, 1], submission[0])
        self.assertEqual([1, 1, 0, 2], submission[1])
        self.assertEqual([1, 9, 0, 3], submission[2])
        self.assertEqual([1, 4, 0, 4], submission[3])
        self.assertEqual([4, 9, 1, 1], submission[4])
        self.assertEqual([4, 1, 0, 2], submission[5])
        self.assertEqual([4, 4, 0, 3], submission[6])
        self.assertEqual([4, 5, 0, 4], submission[7])
        self.assertEqual([3, 4, 1, 1], submission[8])
        self.assertEqual([3, 1, 0, 2], submission[9])
        self.assertEqual([3, 9, 0, 3], submission[10])
        self.assertEqual([3, 5, 0, 4], submission[11])
    
    def test_np_array(self):
        top_n = 3
        classes = np.array(["9", "3", "7"])

        prediction = [
            [0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5]
        ]

        glc_ids = [1, 2]
 
        submission = submission_maker._make_submission(top_n, classes, prediction, glc_ids)
 
        self.assertEqual(3*2, len(submission)) # Anzahl Klassen * Anzahl an Predictions (Größe des Validierungssets)

        ### glc_id,species_glc_id,probability,rank ### 
        self.assertEqual([1, 7, 0.7, 1], submission[0])
        self.assertEqual([1, 3, 0.6, 2], submission[1])
        self.assertEqual([1, 9, 0.5, 3], submission[2])
        self.assertEqual([2, 9, 0.7, 1], submission[3])
        self.assertEqual([2, 3, 0.6, 2], submission[4])
        self.assertEqual([2, 7, 0.5, 3], submission[5])

    def test_df(self):
        top_n = 3
        classes = ["9", "3", "7"]

        prediction = [
            [0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5]
        ]

        glc_ids = [1, 2]
 
        submission_df = submission_maker.make_submission_df(top_n, classes, prediction, glc_ids)
        self.assertEqual(3*2, len(submission_df.index))

        submission_matrix = submission_df.as_matrix()
        self.assertEqual([1, 7, 0.7, 1], list(submission_matrix[0]))
        self.assertEqual([1, 3, 0.6, 2], list(submission_matrix[1]))
        self.assertEqual([1, 9, 0.5, 3], list(submission_matrix[2]))
        self.assertEqual([2, 9, 0.7, 1], list(submission_matrix[3]))
        self.assertEqual([2, 3, 0.6, 2], list(submission_matrix[4]))
        self.assertEqual([2, 7, 0.5, 3], list(submission_matrix[5]))


if __name__ == '__main__':
    unittest.main()