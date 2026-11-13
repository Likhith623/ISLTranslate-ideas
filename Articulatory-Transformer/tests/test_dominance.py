"""
Unit tests for Kinematic Dominance Routing and Torso-Midline Normalization.
"""

import math
import sys
import unittest
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from articulatory_transformer.dominance import (
    compute_hand_energies_pure,
    compute_torso_midline_pure,
)


class TestDominanceAndTorsoMidline(unittest.TestCase):
    def test_torso_midline_centering(self):
        """Verify dynamic shoulder midpoint is centered at (0, 0)."""
        T = 10
        # Create sequence with left shoulder at (10, 20) and right shoulder at (30, 20)
        # Midpoint is (20, 20), shoulder width is 20.0
        seq = []
        for t in range(T):
            frame = [[0.0, 0.0, 0.0] for _ in range(66)]
            frame[42] = [10.0, 20.0, 0.0]  # Left shoulder
            frame[43] = [30.0, 20.0, 0.0]  # Right shoulder
            # Landmark 0 at (20, 20) -> should normalize to (0, 0)
            frame[0] = [20.0, 20.0, 0.0]
            # Landmark 1 at (30, 20) -> should normalize to (0.5, 0)
            frame[1] = [30.0, 20.0, 0.0]
            seq.append(frame)

        norm_seq, widths = compute_torso_midline_pure(seq, left_shoulder_idx=42, right_shoulder_idx=43)

        self.assertEqual(len(norm_seq), T)
        self.assertEqual(len(widths), T)

        for t in range(T):
            # Midpoint point at (20, 20) should normalize to exactly (0, 0)
            self.assertAlmostEqual(norm_seq[t][0][0], 0.0, places=5)
            self.assertAlmostEqual(norm_seq[t][0][1], 0.0, places=5)

            # Left and right shoulders should be symmetric about 0 on x-axis
            x_ls = norm_seq[t][42][0]
            x_rs = norm_seq[t][43][0]
            self.assertAlmostEqual(x_ls + x_rs, 0.0, places=5)
            self.assertAlmostEqual(x_rs - x_ls, 1.0, places=5)

    def test_right_hand_dominance(self):
        """Verify active right hand is correctly classified as dominant."""
        T = 10
        seq = []
        for t in range(T):
            frame = [[0.0, 0.0, 0.0] for _ in range(66)]
            # Left hand static at (0, 0, 0)
            # Right hand moving rapidly: position x = t * 2.0
            for j in range(21, 42):
                frame[j] = [t * 2.0, t * 1.5, 0.0]
            seq.append(frame)

        e_l, e_r, dom = compute_hand_energies_pure(seq)
        self.assertEqual(dom, "right")
        self.assertAlmostEqual(e_l, 0.0)
        self.assertGreater(e_r, 0.0)

    def test_left_hand_dominance(self):
        """Verify active left hand (left-handed signer) is correctly classified as dominant."""
        T = 10
        seq = []
        for t in range(T):
            frame = [[0.0, 0.0, 0.0] for _ in range(66)]
            # Right hand static
            # Left hand moving rapidly
            for j in range(0, 21):
                frame[j] = [t * 3.0, 0.0, 0.0]
            seq.append(frame)

        e_l, e_r, dom = compute_hand_energies_pure(seq)
        self.assertEqual(dom, "left")
        self.assertGreater(e_l, e_r)


if __name__ == "__main__":
    unittest.main()

