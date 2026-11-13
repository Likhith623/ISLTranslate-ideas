"""
AST Structural and Architectural Integrity Tests for AB-ST.
Validates module definitions, classes, mathematical signatures, and config schemas.
"""

import ast
import inspect
import sys
import unittest
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from articulatory_transformer.config import (
    ArticulatorConfig,
    BindingConfig,
    DecoderConfig,
    LossConfig,
    ModelConfig,
    PoseConfig,
    ResamplerConfig,
    TemporalConfig,
    VisionConfig,
)


class TestModelStructure(unittest.TestCase):
    def test_config_defaults_and_validation(self):
        """Verify ModelConfig initializes and passes validation with strict architectural constraints."""
        cfg = ModelConfig()
        cfg.validate()
        
        # Verify key architectural invariants
        self.assertEqual(cfg.pose.num_joints, 66)
        self.assertEqual(cfg.articulator.num_articulators, 4)
        self.assertEqual(cfg.resampler.num_queries, 32)
        self.assertEqual(cfg.binding.hidden_dim, cfg.articulator.embed_dim)
        self.assertEqual(cfg.articulator.embed_dim, cfg.temporal.hidden_dim)
        self.assertEqual(cfg.resampler.output_dim, cfg.decoder.d_model)

    def test_config_dict_roundtrip(self):
        """Verify configuration converts to and from dictionary without loss."""
        cfg = ModelConfig()
        d = cfg.to_dict()
        cfg2 = ModelConfig.from_dict(d)
        self.assertEqual(cfg.pose.left_hand_slice, cfg2.pose.left_hand_slice)
        self.assertEqual(cfg.temporal.conv_kernel_size, cfg2.temporal.conv_kernel_size)
        self.assertEqual(cfg.decoder.pretrained_model_name, cfg2.decoder.pretrained_model_name)

    def test_ast_verification_of_core_modules(self):
        """Parse source code ASTs to verify all architectural modules and classes exist."""
        pkg_root = Path(__file__).resolve().parent.parent / "src" / "articulatory_transformer"
        
        required_classes = {
            "dominance.py": ["TorsoMidlineNormalizer", "KinematicDominanceRouter"],
            "vision.py": ["DINOv2SpatialExtractor", "FeatureCacheManager"],
            "binding.py": ["DeformablePatchBinder"],
            "articulators.py": ["ArticulatorAttentionPooler", "FactoredArticulatorGrouper"],
            "temporal.py": ["ConformerConvModule", "AxialSpatioTemporalBlock", "AxialSpatioTemporalConformer"],
            "resampler.py": ["QFormerLayer", "ArticulatoryQFormer"],
            "decoder.py": ["TranslationDecoderAdapter"],
            "model.py": ["ArticulatoryTransformer"],
            "losses.py": ["CompositeABSTLoss"],
            "data.py": ["SignTranslationDataset"],
        }

        for filename, expected_cls_names in required_classes.items():
            py_file = pkg_root / filename
            self.assertTrue(py_file.exists(), f"File {py_file} does not exist")
            
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))
                
            found_classes = {
                node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            }
            
            for cls_name in expected_cls_names:
                self.assertIn(
                    cls_name,
                    found_classes,
                    f"Class {cls_name} not defined in AST of {filename}",
                )

    def test_metric_accuracy_mathematical_invariants(self):
        """Verify BLEU and ROUGE calculation mathematical invariants."""
        # Add scripts to sys.path to import evaluate
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import evaluate

        # Perfect match
        sent = "a b c d e"
        bleus = evaluate.sentence_bleu(sent, sent)
        rouge = evaluate.sentence_rouge_l(sent, sent)
        self.assertAlmostEqual(bleus[0], 100.0)
        self.assertAlmostEqual(bleus[3], 100.0)
        self.assertAlmostEqual(rouge, 100.0)

        # Complete mismatch
        mismatch_bleu = evaluate.sentence_bleu("x y z", "a b c")
        self.assertAlmostEqual(mismatch_bleu[0], 0.0)
        mismatch_rouge = evaluate.sentence_rouge_l("x y z", "a b c")
        self.assertAlmostEqual(mismatch_rouge, 0.0)

        # Subsequence match
        sub_rouge = evaluate.sentence_rouge_l("a b c", "a b c d e")
        # LCS is 3, len hyp is 3 (prec=1.0), len ref is 5 (rec=0.6). F1 = 2*0.6 / 1.6 = 0.75 * 100 = 75.0
        self.assertAlmostEqual(sub_rouge, 75.0, places=2)


if __name__ == "__main__":
    unittest.main()

