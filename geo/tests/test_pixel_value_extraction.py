import unittest
import numpy as np

from geo.preprocessing.pixel_value_extraction import get_pixel_value

class TestGetPixelValueMethod(unittest.TestCase):
    def test_img_value_1(self):
        img = np.array([
            [0,0,0,0],
            [0,1,4,0],
            [0,2,3,0],
            [0,0,0,0],
        ])

        value = get_pixel_value(img, 1)
        self.assertEqual((1+2+3+4) / 4, value)

    def test_img_value_2(self):
        img = np.array([
            [0,0,0,0],
            [0,1,4,0],
            [0,2,3,0],
            [0,0,0,0],
        ])

        value = get_pixel_value(img, 2)
        self.assertEqual((1+2+3+4) / 16, value)
    

    def test_img_value_big_values_uint8(self):
        arr = []
        for _ in range(64):
            c = []
            for _ in range(64):
                c.append(255)
            arr.append(c)

        img = np.array(arr, dtype=np.uint8)

        value = get_pixel_value(img, 2)
        self.assertEqual(255, value)
    
    def test_img_value_3(self):
        img = np.array([
            [0,0,0,1,0,0],
            [0,1,0,1,0,0],
            [0,0,1,0,1,1],
            [0,0,1,0,0,1],
            [0,0,1,0,0,0],
            [1,0,0,0,0,1],
        ])

        value1 = get_pixel_value(img, 1)
        value2 = get_pixel_value(img, 2)
        value3 = get_pixel_value(img, 3)
        self.assertEqual(2 / 4, value1)
        self.assertEqual(6 / 16, value2)
        self.assertEqual(11 / 36, value3)

if __name__ == '__main__':
    unittest.main()