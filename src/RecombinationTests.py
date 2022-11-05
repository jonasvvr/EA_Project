import unittest
import hamilton_cycle as hc


class Test(unittest.TestCase):

    def test_appendSS(self):
        allSS = []
        SS_empty = []
        SS_len_1 = [1]
        SS_len_2 = [1, 2]
        SS_len_3 = [1, 2, 3]

        hc.appendSS(allSS, SS_empty)
        self.assertEqual(len(allSS), 0)

        hc.appendSS(allSS, SS_len_1)
        self.assertEqual(len(allSS), 0)

        hc.appendSS(allSS, SS_len_2)
        self.assertEqual(allSS, [[1, 2]])

        hc.appendSS(allSS, SS_len_3)
        self.assertEqual(allSS, [[1, 2, 3]])


if __name__ == '__main__':
    unittest.main()