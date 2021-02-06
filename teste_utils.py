import math
from utils import *
import unittest
import pandas_datareader as data_reader

class MyTest(unittest.TestCase):
    def test_sigmoid(self):
        self.assertEqual(sigmoid(1.4), 0.8021838885585817)
        self.assertAlmostEqual(sigmoid(1.4), 0.80218388856)
    
    def test_sigmoid_neg(self):
        self.assertNotEqual(sigmoid(1.4), 0.9)

    def test_price_format(self):
        pos_right_format = "R$ 3.96"
        neg_right_format = "-R$ 3.96"
        self.assertEqual(stocks_price_format(3.958677), pos_right_format)
        self.assertEqual(stocks_price_format(-3.958677), neg_right_format)

    def test_price_format_neg(self):
        pos_right_format = "R$ 3.959"
        neg_right_format = "-R$ 3.959"
        self.assertNotEqual(stocks_price_format(3.958677), pos_right_format)
        self.assertNotEqual(stocks_price_format(-3.958677), neg_right_format)


    def test_dataset_loader(self):
        stock_name = "MSFT"
        self.assertIsNotNone(dataset_loader(stock_name), None)
        

    def test_state_creator(self):
        data = dataset_loader("MSFT")
        self.assertIsNotNone(state_creator(data, 10,4), None)
        


if __name__ == '__main__':
    unittest.main()