"""
Unit tests for Canonical 543 -> 66 MediaPipe Landmark Adapter.
"""

import math
import sys
import unittest
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from articulatory_transformer.adapter_543_to_66 import (
    BODY_RANGE,
    CANONICAL_66_INDICES_IN_543,
    FACE_RANGE,
    HOLISTIC_LEFT_HAND_OFFSET,
    HOLISTIC_RIGHT_HAND_OFFSET,
    LEFT_HAND_RANGE,
    RIGHT_HAND_RANGE,
    map_543_to_66,
    map_holistic_543_to_canonical_66_list,
    validate_canonical_66_structure,
)


class TestAdapter543To66(unittest.TestCase):
    def test_canonical_slices_and_counts(self):
        """Verify partition slices total exactly 66 landmarks with zero gaps or overlaps."""
        self.assertEqual(LEFT_HAND_RANGE, (0, 21))
        self.assertEqual(RIGHT_HAND_RANGE, (21, 42))
        self.assertEqual(BODY_RANGE, (42, 48))
        self.assertEqual(FACE_RANGE, (48, 66))

        lh_count = LEFT_HAND_RANGE[1] - LEFT_HAND_RANGE[0]
        rh_count = RIGHT_HAND_RANGE[1] - RIGHT_HAND_RANGE[0]
        bd_count = BODY_RANGE[1] - BODY_RANGE[0]
        fc_count = FACE_RANGE[1] - FACE_RANGE[0]

        self.assertEqual(lh_count, 21)
        self.assertEqual(rh_count, 21)
        self.assertEqual(bd_count, 6)
        self.assertEqual(fc_count, 18)
        self.assertEqual(lh_count + rh_count + bd_count + fc_count, 66)
        self.assertEqual(len(CANONICAL_66_INDICES_IN_543), 66)

    def test_hand_and_body_index_sources(self):
        """Verify the exact source landmark IDs from Holistic 543."""
        # Left hand: 501..521
        for i in range(21):
            expected_src = HOLISTIC_LEFT_HAND_OFFSET + i
            self.assertEqual(CANONICAL_66_INDICES_IN_543[i], expected_src)

        # Right hand: 522..542
        for i in range(21):
            expected_src = HOLISTIC_RIGHT_HAND_OFFSET + i
            self.assertEqual(CANONICAL_66_INDICES_IN_543[21 + i], expected_src)

        # Body: [11, 12, 13, 14, 15, 16]
        expected_body = [11, 12, 13, 14, 15, 16]
        self.assertEqual(CANONICAL_66_INDICES_IN_543[42:48], expected_body)

    def test_single_frame_mapping_exactness(self):
        """Verify exact coordinate mapping without distortion or offset shifts."""
        # Create synthetic 543 landmarks where each coordinate equals its landmark ID
        raw_543 = [[float(i), float(i) * 2.0, float(i) * 3.0] for i in range(543)]
        
        mapped_66 = map_holistic_543_to_canonical_66_list(raw_543)
        self.assertEqual(len(mapped_66), 66)

        # Check left shoulder (canonical index 42 -> raw index 11)
        self.assertEqual(mapped_66[42], [11.0, 22.0, 33.0])
        # Check right shoulder (canonical index 43 -> raw index 12)
        self.assertEqual(mapped_66[43], [12.0, 24.0, 36.0])

        # Check left wrist (canonical index 46 -> raw index 15)
        self.assertEqual(mapped_66[46], [15.0, 30.0, 45.0])

        # Check left hand wrist (canonical index 0 -> raw index 501)
        self.assertEqual(mapped_66[0], [501.0, 1002.0, 1503.0])

        # Check right hand wrist (canonical index 21 -> raw index 522)
        self.assertEqual(mapped_66[21], [522.0, 1044.0, 1566.0])

    def test_sequence_mapping(self):
        """Verify sequence mapping [T, 543, 3] -> [T, 66, 3]."""
        T = 5
        seq_543 = [[[float(i + t), float(i), 0.0] for i in range(543)] for t in range(T)]
        seq_66 = map_543_to_66(seq_543)

        self.assertEqual(len(seq_66), T)
        for t in range(T):
            self.assertEqual(len(seq_66[t]), 66)
            self.assertEqual(seq_66[t][42][0], 11.0 + t)

    def test_validation_report(self):
        """Verify data integrity checks catch NaNs and small shoulder distances."""
        # Valid frame
        valid_frame = [[float(i), float(i), 0.0] for i in range(66)]
        valid_frame[42] = [0.0, 0.0, 0.0]
        valid_frame[43] = [0.5, 0.0, 0.0]
        rep = validate_canonical_66_structure(valid_frame)
        self.assertTrue(rep["valid"])
        self.assertAlmostEqual(rep["shoulder_dist"], 0.5)

        # Frame with NaN
        nan_frame = [list(pt) for pt in valid_frame]
        nan_frame[5][0] = float("nan")
        rep_nan = validate_canonical_66_structure(nan_frame)
        self.assertFalse(rep_nan["valid"])
        self.assertTrue(any("NaN" in err for err in rep_nan["errors"]))


if __name__ == "__main__":
    unittest.main()

