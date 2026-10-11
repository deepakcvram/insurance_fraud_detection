"""
ClaimShield AI — Phase 5: Historical Analytics & Batch Inference Pipeline
==========================================================================
Connects to HDFS, reads historical validated claim records in JSONL format,
validates and deduplicates records, transforms features via FeaturePreprocessor,
runs batch inference using model.pkl, computes summary analytics, and stores
auditable results.

Usage:
------
    # Run historical batch analysis for all claims in HDFS:
    python kafka_pipeline/historical_analytics.py

    # Override fraud decision threshold via CLI or environment:
    python kafka_pipeline/historical_analytics.py --threshold 0.65

    # Run on a specific historical HDFS file:
    python kafka_pipeline/historical_analytics.py --file claims_20261011.jsonl
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kafka_pipeline.config import KafkaConfig
from kafka_pipeline.hdfs_storage import HDFSStorage, HDFSError
from kafka_pipeline.feature_preprocessor import FeaturePreprocessor, FeaturePreprocessingError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("claimshield.analytics")


class HistoricalAnalyticsPipeline:
    """
    Historical claims analysis engine integrating HDFS ingestion,
    preprocessing, XGBoost inference, and analytics reporting.
    """

    def __init__(self, cfg: Optional[KafkaConfig] = None):
        self.cfg = cfg or KafkaConfig()
        self.threshold = self.cfg.fraud_threshold
        self.output_dir = self.cfg.analytics_output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize HDFS client
        self.hdfs = HDFSStorage(self.cfg)

        # Initialize Feature Preprocessor
        self.preprocessor = FeaturePreprocessor(cfg=self.cfg)

        # Load trained XGBoost model
        self._load_model()

    def _load_model(self) -> None:
        """Load model.pkl safely."""
        if not os.path.exists(self.cfg.model_path):
            raise FileNotFoundError(f"Model file not found: {self.cfg.model_path}")

        with open(self.cfg.model_path, "rb") as f:
            self.model = pickle.load(f)
        log.info("Model loaded successfully: %s", type(self.model).__name__)

    def ingest_from_hdfs(self, target_file: Optional[str] = None) -> dict[str, Any]:
        """
        Step 1: Ingest historical claim records from HDFS.

        Returns
        -------
        Dictionary containing:
        - 'accepted': list of valid deduplicated raw claim dicts
        - 'rejected': list of dicts with rejection reason
        - 'stats': ingestion statistics (read, accepted, rejected, deduplicated)
        """
        log.info("Connecting to HDFS at %s...", self.cfg.hdfs_namenode_url)
        if not self.hdfs.is_available():
            raise HDFSError(f"HDFS service unavailable at {self.cfg.hdfs_namenode_url}")

        raw_records: list[dict[str, Any]] = []

        if target_file:
            files_to_read = [target_file]
        else:
            all_files = self.hdfs.list_files()
            files_to_read = [f for f in all_files if f.endswith(".jsonl")]
            if not files_to_read:
                log.warning("No JSONL claim files found in HDFS directory %s", self.cfg.hdfs_claims_path)

        for fname in files_to_read:
            try:
                records = self.hdfs.read_claims(fname)
                log.info("Read %d records from HDFS file %s", len(records), fname)
                raw_records.extend(records)
            except Exception as exc:
                log.error("Failed to read HDFS file %s: %s", fname, exc)

        total_read = len(raw_records)
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen_claim_ids: set[str] = set()
        deduplicated_count = 0

        for idx, rec in enumerate(raw_records):
            if not isinstance(rec, dict):
                rejected.append({
                    "record_index": idx,
                    "claim_id": "<unknown>",
                    "reason": f"Malformed non-dictionary record of type {type(rec).__name__}",
                })
                continue

            claim_id = rec.get("claim_id")
            if not claim_id:
                rejected.append({
                    "record_index": idx,
                    "claim_id": "<missing>",
                    "reason": "Missing required field 'claim_id'",
                })
                continue

            # Deduplication check
            if claim_id in seen_claim_ids:
                deduplicated_count += 1
                rejected.append({
                    "record_index": idx,
                    "claim_id": claim_id,
                    "reason": "Duplicate claim_id ignored",
                })
                continue

            # Schema and field validation
            validation_errors = self.preprocessor.validate_raw_record(rec)
            if validation_errors:
                rejected.append({
                    "record_index": idx,
                    "claim_id": claim_id,
                    "reason": "; ".join(validation_errors),
                })
                continue

            seen_claim_ids.add(claim_id)
            accepted.append(rec)

        stats = {
            "total_read": total_read,
            "accepted": len(accepted),
            "rejected": len(rejected) - deduplicated_count,
            "deduplicated": deduplicated_count,
        }

        log.info(
            "HDFS Ingestion Complete: %d read | %d accepted | %d rejected | %d duplicates",
            stats["total_read"], stats["accepted"], stats["rejected"], stats["deduplicated"]
        )

        return {"accepted": accepted, "rejected": rejected, "stats": stats}

    def run_inference(self, claims: list[dict[str, Any]], threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Step 2 & 3: Transform features and run batch inference using model.pkl.

        Returns
        -------
        DataFrame with claim identifiers, probabilities, predictions, and audit columns.
        """
        eval_threshold = threshold if threshold is not None else self.threshold

        if not claims:
            log.warning("Empty claims list provided for inference.")
            return pd.DataFrame(columns=[
                "claim_id", "submitted_at", "fraud_probability", "predicted_class",
                "fraud_flag", "decision_threshold", "status"
            ])

        log.info("Transforming %d valid claims into 86 model features...", len(claims))
        df_features = self.preprocessor.transform_batch(claims)

        # Batch prediction with XGBoost model
        log.info("Running batch inference using %s...", type(self.model).__name__)
        probs = self.model.predict_proba(df_features)[:, 1]  # class 1 = Fraud
        predictions = (probs >= eval_threshold).astype(int)

        results = []
        for idx, claim in enumerate(claims):
            prob = float(probs[idx])
            pred_class = int(predictions[idx])
            results.append({
                "claim_id": claim.get("claim_id"),
                "submitted_at": claim.get("submitted_at"),
                "Make": claim.get("Make"),
                "VehicleCategory": claim.get("VehicleCategory"),
                "VehiclePrice": claim.get("VehiclePrice"),
                "Age": claim.get("Age"),
                "Fault": claim.get("Fault"),
                "PolicyType": claim.get("PolicyType"),
                "Deductible": claim.get("Deductible"),
                "PastNumberOfClaims": claim.get("PastNumberOfClaims"),
                "fraud_probability": round(prob, 4),
                "predicted_class": pred_class,
                "fraud_flag": "FLAGGED_FOR_INVESTIGATION" if pred_class == 1 else "STANDARD_PROCESSING",
                "decision_threshold": eval_threshold,
                "status": "INFERRED_SUCCESS",
            })

        df_results = pd.DataFrame(results)
        log.info(
            "Inference completed: %d claims processed (%d flagged as potential fraud at threshold %.2f)",
            len(df_results), (df_results["predicted_class"] == 1).sum(), eval_threshold
        )
        return df_results

    def compute_analytics(self, df_results: pd.DataFrame, rejection_records: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Step 4: Compute historical statistical summaries using Pandas.
        """
        if df_results.empty:
            return {
                "total_claims": 0,
                "flagged_claims": 0,
                "flagged_percentage": 0.0,
                "mean_fraud_probability": 0.0,
                "rejections_count": len(rejection_records),
                "distribution": {},
                "summary": "No claims available to compute analytics."
            }

        total = len(df_results)
        flagged = int((df_results["predicted_class"] == 1).sum())
        flagged_pct = round((flagged / total) * 100, 2)
        mean_prob = round(float(df_results["fraud_probability"].mean()), 4)
        median_prob = round(float(df_results["fraud_probability"].median()), 4)
        max_prob = round(float(df_results["fraud_probability"].max()), 4)
        min_prob = round(float(df_results["fraud_probability"].min()), 4)

        # Risk distribution bands
        low_risk = int((df_results["fraud_probability"] < 0.30).sum())
        medium_risk = int(((df_results["fraud_probability"] >= 0.30) & (df_results["fraud_probability"] < 0.70)).sum())
        high_risk = int((df_results["fraud_probability"] >= 0.70).sum())

        # Groupings by claim attributes
        make_breakdown = df_results.groupby("Make")["predicted_class"].agg(["count", "sum"]).to_dict("index")
        category_breakdown = df_results.groupby("VehicleCategory")["predicted_class"].agg(["count", "sum"]).to_dict("index")
        fault_breakdown = df_results.groupby("Fault")["predicted_class"].agg(["count", "sum"]).to_dict("index")

        analytics_summary = {
            "total_claims_processed": total,
            "flagged_for_investigation": flagged,
            "flagged_percentage": flagged_pct,
            "decision_threshold": self.threshold,
            "probability_statistics": {
                "mean": mean_prob,
                "median": median_prob,
                "min": min_prob,
                "max": max_prob,
            },
            "risk_tier_distribution": {
                "low_risk_under_30pct": low_risk,
                "medium_risk_30_to_70pct": medium_risk,
                "high_risk_over_70pct": high_risk,
            },
            "attribute_breakdowns": {
                "by_make": make_breakdown,
                "by_vehicle_category": category_breakdown,
                "by_fault": fault_breakdown,
            },
            "ingestion_rejections": len(rejection_records),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        return analytics_summary

    def save_results(
        self,
        df_results: pd.DataFrame,
        rejected: list[dict[str, Any]],
        analytics: dict[str, Any],
    ) -> dict[str, str]:
        """
        Step 5: Write auditable outputs to analytics_output/ and optionally HDFS.
        Safe to re-run: uses deterministic filenames without modifying raw HDFS files.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

        # 1. Save detailed claims scored results (CSV & JSON)
        results_csv = os.path.join(self.output_dir, f"batch_inference_results_{date_str}.csv")
        df_results.to_csv(results_csv, index=False)

        # 2. Save rejected / audit error records
        rejections_json = os.path.join(self.output_dir, f"rejected_claims_{date_str}.json")
        with open(rejections_json, "w", encoding="utf-8") as f:
            json.dump(rejected, f, indent=2)

        # 3. Save analytical summary report
        summary_json = os.path.join(self.output_dir, f"historical_analytics_report_{date_str}.json")
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(analytics, f, indent=2)

        # 4. Save high-risk investigation queue
        investigation_queue_csv = os.path.join(self.output_dir, "investigation_queue.csv")
        high_risk_df = df_results[df_results["predicted_class"] == 1]
        high_risk_df.to_csv(investigation_queue_csv, index=False)

        # 5. Persist analytics report to HDFS for persistent auditing
        hdfs_report_path = f"{self.cfg.hdfs_analytics_path}/report_{date_str}.json"
        try:
            self.hdfs.ensure_directory(self.cfg.hdfs_analytics_path)
            report_payload = json.dumps(analytics, indent=2).encode("utf-8")
            self.hdfs.append_or_create(hdfs_report_path, report_payload)
            log.info("Persisted analytics report to HDFS at '%s'", hdfs_report_path)
        except Exception as exc:
            log.warning("Could not persist analytics report to HDFS (local saved): %s", exc)

        output_files = {
            "results_csv": results_csv,
            "rejections_json": rejections_json,
            "summary_json": summary_json,
            "investigation_queue_csv": investigation_queue_csv,
            "hdfs_report_path": hdfs_report_path,
        }

        log.info("Batch results saved successfully to '%s'", self.output_dir)
        return output_files

    def run(self, target_file: Optional[str] = None, threshold: Optional[float] = None) -> dict[str, Any]:
        """Execute the end-to-end historical analytics pipeline."""
        if threshold is not None:
            self.threshold = threshold

        log.info("Starting Phase 5 Historical Analytics Pipeline (Threshold: %.2f)...", self.threshold)
        start_time = time.time()

        # Step 1: Ingest & Deduplicate
        ingestion = self.ingest_from_hdfs(target_file)
        accepted_claims = ingestion["accepted"]
        rejected_claims = ingestion["rejected"]

        # Step 2 & 3: Preprocess & Inference
        df_results = self.run_inference(accepted_claims, self.threshold)

        # Step 4: Compute Analytics
        analytics = self.compute_analytics(df_results, rejected_claims)

        # Step 5: Save Outputs
        output_files = self.save_results(df_results, rejected_claims, analytics)

        elapsed = round(time.time() - start_time, 2)
        log.info("Historical Analytics Pipeline completed in %.2fs", elapsed)

        return {
            "ingestion_stats": ingestion["stats"],
            "analytics_summary": analytics,
            "output_files": output_files,
            "elapsed_seconds": elapsed,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ClaimShield AI — Phase 5: Historical Analytics Pipeline")
    parser.add_argument("--threshold", type=float, default=None, help="Fraud decision probability threshold (default: 0.70)")
    parser.add_argument("--file", type=str, default=None, help="Specific HDFS JSONL claims file name to analyze")
    args = parser.parse_args()

    cfg = KafkaConfig()
    pipeline = HistoricalAnalyticsPipeline(cfg)
    results = pipeline.run(target_file=args.file, threshold=args.threshold)

    print("\n" + "=" * 60)
    print(" CLAIMSHIELD AI — PHASE 5 HISTORICAL ANALYTICS REPORT ")
    print("=" * 60)
    stats = results["ingestion_stats"]
    print(f"Total Claims Read:         {stats['total_read']}")
    print(f"Accepted Claims:           {stats['accepted']}")
    print(f"Rejected / Corrupt Claims: {stats['rejected']}")
    print(f"Deduplicated Claims:       {stats['deduplicated']}")
    print("-" * 60)
    summary = results["analytics_summary"]
    print(f"Decision Threshold:        {summary.get('decision_threshold')}")
    print(f"Flagged for Investigation: {summary.get('flagged_for_investigation')} ({summary.get('flagged_percentage')}%)")
    probs = summary.get("probability_statistics", {})
    print(f"Mean Fraud Probability:    {probs.get('mean')}")
    print(f"Max Fraud Probability:     {probs.get('max')}")
    tiers = summary.get("risk_tier_distribution", {})
    print(f"High Risk Claims (>=0.70): {tiers.get('high_risk_over_70pct')}")
    print(f"Medium Risk (0.30 - 0.70): {tiers.get('medium_risk_30_to_70pct')}")
    print(f"Low Risk (< 0.30):         {tiers.get('low_risk_under_30pct')}")
    print("-" * 60)
    print("Output Files Generated:")
    for k, v in results["output_files"].items():
        print(f"  - {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
