import top_k_acc
import unittest

class TestGetRanksMethod(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()