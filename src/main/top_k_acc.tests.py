import top_k_acc
import unittest

class TestGetRanksMethod(unittest.TestCase):
    '''Class for tests of the method top_k_accuracy() of the module top_k_acc.'''
    def test_basic(self):
        y_pred = [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ]

        class_map = [1,2,3,4]

        y_true = [1,2,3,4]

        self.assertEqual(0.25, top_k_acc.top_k_acc(y_pred, y_true, class_map, 1))
        self.assertEqual(0.5, top_k_acc.top_k_acc(y_pred, y_true, class_map, 2))
        self.assertEqual(0.75, top_k_acc.top_k_acc(y_pred, y_true, class_map, 3))
        self.assertEqual(1, top_k_acc.top_k_acc(y_pred, y_true, class_map, 4))
 
    def test_multicore(self):
        y_pred = [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ]

        class_map = [1,2,3,4]

        y_true = [1,2,3,4]
        t = top_k_acc.top_k_accuracy()
        self.assertEqual(0.25, t.get_score(y_pred, y_true, class_map, 1))
        self.assertEqual(0.5, t.get_score(y_pred, y_true, class_map, 2))
        self.assertEqual(0.75, t.get_score(y_pred, y_true, class_map, 3))
        self.assertEqual(1, t.get_score(y_pred, y_true, class_map, 4))

    def test_extrem(self):
        l = 1000000
        y_pred = [[0.1, 0.2, 0.3, 0.4]  for _ in range(l)]

        class_map = [x for x in range(1,4)]
        y_true = [4 for x in range(l)]
        top_k_acc.top_k_acc(y_pred, y_true, class_map, 1)
        
    def test_extrem_multicore(self):
        l = 1000000
        y_pred = [[0.1, 0.2, 0.3, 0.4]  for _ in range(l)]

        class_map = [x for x in range(1,4)]
        y_true = [4 for x in range(l)]
        t = top_k_acc.top_k_accuracy()

        self.assertEqual(1, t.get_score(y_pred, y_true, class_map, 1))

if __name__ == '__main__':
    unittest.main()