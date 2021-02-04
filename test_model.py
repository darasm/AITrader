from model import AITRADE
import unittest


class TestAITRADE(unittest.TestCase):
    def test_trade(self):
        obj = AITRADE()
        self.assertEqual(obj.trade(6), any() )
        
    
    def test_batch_train(self):
        obj = AITRADE()
        self.assertEqual(obj.batch_train(5), any())
        