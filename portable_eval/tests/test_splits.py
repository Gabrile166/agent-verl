import unittest

from agent_eval.sciworld_eval import iter_l1_test_variations


class ScienceWorldSplitTests(unittest.TestCase):
    def test_l1_test_split_is_complete_and_unique(self):
        pairs = list(iter_l1_test_variations())
        self.assertEqual(len(pairs), 1684)
        self.assertEqual(len(set(pairs)), 1684)
        self.assertEqual(set(task for task, _ in pairs), set(range(29)))

    def test_l1_uses_held_out_tail_variations(self):
        pairs = set(iter_l1_test_variations())
        self.assertIn((0, 21), pairs)
        self.assertIn((0, 29), pairs)
        self.assertNotIn((0, 20), pairs)
        self.assertIn((15, 1038), pairs)
        self.assertIn((15, 1385), pairs)


if __name__ == "__main__":
    unittest.main()
