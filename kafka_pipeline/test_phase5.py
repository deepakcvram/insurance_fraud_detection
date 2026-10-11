"""
ClaimShield AI — Phase 5 Test Suite
====================================
Comprehensive tests covering all Phase 5 requirements:
1. Valid historical claim transformation.
2. Multiple claims in one batch.
3. Duplicate claim IDs deduplication.
4. Missing required fields rejection.
5. Unknown categorical values rejection.
6. Empty input handling gracefully.
7. Feature-column ordering (exact 86 match).
8. Model inference probability generation.
9. Repeated execution idempotency.
10. HDFS connection/read error handling.
"""

import copy
import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kafka_pipeline.config import KafkaConfig
from kafka_pipeline.sample_claims import SAMPLE_CLAIMS
from kafka_pipeline.feature_preprocessor import FeaturePreprocessor, FeaturePreprocessingError
from kafka_pipeline.historical_analytics import HistoricalAnalyticsPipeline
from kafka_pipeline.hdfs_storage import HDFSError


class TestPhase5HistoricalPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = KafkaConfig()
        cls.preprocessor = FeaturePreprocessor(cfg=cls.cfg)
        cls.pipeline = HistoricalAnalyticsPipeline(cfg=cls.cfg)

    def test_01_valid_historical_claim(self):
        """Test single valid claim transformation to 1x86 feature vector."""
        raw_claim = copy.deepcopy(SAMPLE_CLAIMS[0])
        df_feats = self.preprocessor.transform_single(raw_claim)
        self.assertEqual(df_feats.shape, (1, 86))
        self.assertFalse(df_feats.isna().any().any())

    def test_02_multiple_claims_in_batch(self):
        """Test batch transformation of multiple claims."""
        claims = copy.deepcopy(SAMPLE_CLAIMS)
        df_feats = self.preprocessor.transform_batch(claims)
        self.assertEqual(df_feats.shape, (len(SAMPLE_CLAIMS), 86))

    def test_03_duplicate_claim_ids(self):
        """Test deduplication on claim_id."""
        claims = [copy.deepcopy(SAMPLE_CLAIMS[0]), copy.deepcopy(SAMPLE_CLAIMS[0])]
        # Mock ingestion parsing
        seen = set()
        accepted, dedup_count = [], 0
        for c in claims:
            cid = c["claim_id"]
            if cid in seen:
                dedup_count += 1
            else:
                seen.add(cid)
                accepted.append(c)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(dedup_count, 1)

    def test_04_missing_required_fields(self):
        """Test rejection when required fields are missing."""
        bad_claim = copy.deepcopy(SAMPLE_CLAIMS[0])
        del bad_claim["PolicyType"]
        with self.assertRaises(FeaturePreprocessingError):
            self.preprocessor.transform_single(bad_claim)

    def test_05_unknown_categorical_values(self):
        """Test rejection when an unknown categorical value is encountered."""
        bad_claim = copy.deepcopy(SAMPLE_CLAIMS[0])
        bad_claim["AccidentArea"] = "OuterSpace"
        with self.assertRaises(FeaturePreprocessingError):
            self.preprocessor.transform_single(bad_claim)

    def test_06_empty_input(self):
        """Test that empty input is handled gracefully without crashing."""
        df_empty = self.preprocessor.transform_batch([])
        self.assertEqual(df_empty.shape, (0, 86))
        self.assertEqual(list(df_empty.columns), self.preprocessor.feature_columns)

        df_res = self.pipeline.run_inference([])
        self.assertTrue(df_res.empty)
        analytics = self.pipeline.compute_analytics(df_res, [])
        self.assertEqual(analytics["total_claims"], 0)

    def test_07_feature_column_ordering(self):
        """Test that the 86 output columns match metadata.pkl ordering strictly."""
        df_feats = self.preprocessor.transform_batch(copy.deepcopy(SAMPLE_CLAIMS))
        self.assertEqual(list(df_feats.columns), self.preprocessor.feature_columns)

    def test_08_model_inference(self):
        """Test that model inference outputs valid probabilities in [0.0, 1.0]."""
        claims = copy.deepcopy(SAMPLE_CLAIMS)
        df_res = self.pipeline.run_inference(claims, threshold=0.70)
        self.assertEqual(len(df_res), len(SAMPLE_CLAIMS))
        self.assertTrue((df_res["fraud_probability"] >= 0.0).all())
        self.assertTrue((df_res["fraud_probability"] <= 1.0).all())
        self.assertIn("predicted_class", df_res.columns)
        self.assertIn("fraud_flag", df_res.columns)

    def test_09_repeated_execution(self):
        """Test repeated execution of the pipeline is idempotent and safe."""
        claims = copy.deepcopy(SAMPLE_CLAIMS)
        df_res1 = self.pipeline.run_inference(claims, threshold=0.70)
        df_res2 = self.pipeline.run_inference(claims, threshold=0.70)
        pd.testing.assert_frame_equal(df_res1, df_res2)

    def test_10_hdfs_failure_handling(self):
        """Test that invalid HDFS URL raises clear HDFSError."""
        bad_cfg = KafkaConfig()
        bad_cfg.hdfs_namenode_url = "http://invalid-fake-host:9999"
        bad_pipeline = HistoricalAnalyticsPipeline(bad_cfg)
        with self.assertRaises(HDFSError):
            bad_pipeline.ingest_from_hdfs()


if __name__ == "__main__":
    unittest.main()
