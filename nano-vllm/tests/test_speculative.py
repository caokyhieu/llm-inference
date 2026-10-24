import unittest

from nanovllm.engine.speculative import stop_at_eos, verify_draft
from nanovllm.sampling_params import SamplingParams


class VerificationTests(unittest.TestCase):
    def test_first_rejection(self):
        self.assertEqual(verify_draft([1, 2, 3], [9, 2, 3, 4]), ([9], 0))

    def test_middle_rejection_discards_later_matches(self):
        self.assertEqual(verify_draft([1, 2, 3], [1, 9, 3, 4]), ([1, 9], 1))

    def test_all_accepted_includes_bonus(self):
        self.assertEqual(verify_draft([1, 2, 3], [1, 2, 3, 4]), ([1, 2, 3, 4], 3))

    def test_no_proposals_is_ordinary_target_decoding(self):
        self.assertEqual(verify_draft([], [7]), ([7], 0))

    def test_missing_bonus_is_an_error(self):
        with self.assertRaises(ValueError):
            verify_draft([1], [1])

    def test_eos_truncates_even_inside_an_accepted_prefix(self):
        emitted, _ = verify_draft([1, 99, 3], [1, 99, 3, 4])
        self.assertEqual(stop_at_eos(emitted, 99, False), [1, 99])
        self.assertEqual(stop_at_eos(emitted, 99, True), [1, 99, 3, 4])

    def test_greedy_and_positive_token_budget(self):
        self.assertEqual(SamplingParams(temperature=0).temperature, 0)
        with self.assertRaises(AssertionError):
            SamplingParams(temperature=-1)
        with self.assertRaises(AssertionError):
            SamplingParams(max_tokens=0)


if __name__ == "__main__":
    unittest.main()
