'''
assertEqual(a, b): 
    a와 b가 동일한지 확인합니다.
assertNotEqual(a, b): 
    a와 b가 서로 다른지 확인합니다.
assertTrue(x): 
    x가 True인지 확인합니다.
assertFalse(x): 
    x가 False인지 확인합니다.
assertIs(a, b): 
    a와 b가 같은 객체인지 확인합니다.
assertIsNot(a, b): 
    a와 b가 다른 객체인지 확인합니다.
assertIsNone(x): 
    x가 None인지 확인합니다.
assertIsNotNone(x): 
    x가 None이 아닌지 확인합니다.
assertIn(a, b): 
    a가 b에 속하는지 확인합니다.
assertNotIn(a, b): 
    a가 b에 속하지 않는지 확인합니다.
assertAlmostEqual(a, b): 
    a와 b가 거의 같은지 확인합니다.
assertNotAlmostEqual(a, b): 
    a와 b가 거의 다른지 확인합니다.
assertGreater(a, b): 
    a가 b보다 큰지 확인합니다.
assertGreaterEqual(a, b): 
    a가 b보다 크거나 같은지 확인합니다.
assertLess(a, b): 
    a가 b보다 작은지 확인합니다.
assertLessEqual(a, b): 
    a가 b보다 작거나 같은지 확인합니다.
'''

import unittest
from handle_transformer import HandleTransformer

class TestAddFunction(unittest.TestCase):
    def test1(self):
        def execute():
            HandleTransformer().get_positional_encoding('bert-base-uncased', 10)

        # debug
        execute()

        # test
        self.assertEqual(lambda x: execute(), 5)

if __name__ == '__main__':
    unittest.main()
