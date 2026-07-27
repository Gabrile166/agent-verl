import hashlib
import json
import unittest

from agent_eval.sciworld_eval import iter_l1_test_variations


class ScienceWorldSplitChecksumTests(unittest.TestCase):
    def test_pair_set_matches_original_repository_split(self):
        encoded = json.dumps(
            sorted(iter_l1_test_variations()),
            separators=(",", ":"),
        ).encode()
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            "1108e8ad112f733e8c45106c6521910f649bc342fd24f500413ba9f6562df136",
        )


if __name__ == "__main__":
    unittest.main()
