from Data import Data
import unittest
import numpy as np

class TestEvaluationRanksMethod(unittest.TestCase):
    def test_img_value_1(self):
        data = Data()

        img = np.array([
            [0,0,0,0],
            [0,1,4,0],
            [0,2,3,0],
            [0,0,0,0],
        ])

        value = data.get_image_value(img, 1)
        self.assertEqual((1+2+3+4) / 4, value)

    def test_img_value_2(self):
        data = Data()

        img = np.array([
            [0,0,0,0],
            [0,1,4,0],
            [0,2,3,0],
            [0,0,0,0],
        ])

        value = data.get_image_value(img, 2)
        self.assertEqual((1+2+3+4) / 16, value)
    
    def test_img_value_3(self):
        data = Data()

        img = np.array([
            [0,0,0,1,0,0],
            [0,1,0,1,0,0],
            [0,0,1,0,1,1],
            [0,0,1,0,0,1],
            [0,0,1,0,0,0],
            [1,0,0,0,0,1],
        ])

        value1 = data.get_image_value(img, 1)
        value2 = data.get_image_value(img, 2)
        value3 = data.get_image_value(img, 3)
        self.assertEqual(2 / 4, value1)
        self.assertEqual(6 / 16, value2)
        self.assertEqual(11 / 36, value3)
                    
if __name__ == '__main__':
    unittest.main()