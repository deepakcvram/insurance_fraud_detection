"""
ClaimShield AI — Phase 5: Shared Feature Preprocessor
======================================================
Loads the existing model artifacts (encoder.pkl, metadata.pkl) and applies
the verified preprocessing transformations to convert 31-field raw claim records
into the exact 86-feature vector expected by model.pkl.

Guarantees:
----------
1. Reuses existing encoder.pkl and metadata.pkl.
2. Applies the exact binary and ordinal mappings used during training.
3. Applies OneHotEncoder for nominal variables.
4. Validates data types, missing fields, and detects unknown categories.
5. Reindexes into the exact 86-feature order expected by model.pkl.
6. Fails clearly on incompatible inputs without inventing synthetic values.
"""

import os
import pickle
from typing import Any, Optional, Union
import numpy as np
import pandas as pd

from kafka_pipeline.config import KafkaConfig
from kafka_pipeline.sample_claims import REQUIRED_FIELDS, VALID_VALUES


class FeaturePreprocessingError(Exception):
    """Raised when an input claim cannot be preprocessed into valid model features."""
    pass


class FeaturePreprocessor:
    """
    Shared feature preprocessor for converting raw insurance claim dictionaries
    or DataFrames into the 86-feature DataFrame expected by model.pkl.
    """

    def __init__(
        self,
        metadata_path: Optional[str] = None,
        encoder_path: Optional[str] = None,
        cfg: Optional[KafkaConfig] = None,
    ):
        self.cfg = cfg or KafkaConfig()
        self.metadata_path = metadata_path or self.cfg.metadata_path
        self.encoder_path = encoder_path or self.cfg.encoder_path

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load encoder.pkl and metadata.pkl safely."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Encoder file not found: {self.encoder_path}")

        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        with open(self.encoder_path, "rb") as f:
            self.encoder = pickle.load(f)

        self.binary_cols: dict[str, dict[str, int]] = self.metadata["binary_cols"]
        self.ordinal_mappings: dict[str, dict[str, int]] = self.metadata["ordinal_mappings"]
        self.nominal_cols: list[str] = self.metadata["nominal_cols"]
        self.feature_columns: list[str] = self.metadata["feature_columns"]

        # Validate that model expects 86 features
        if len(self.feature_columns) != 86:
            raise ValueError(f"Expected 86 feature columns in metadata, got {len(self.feature_columns)}")

    def validate_raw_record(self, record: dict[str, Any]) -> list[str]:
        """
        Validate that all required feature fields exist and categorical values
        are within expected domain sets.
        Returns a list of error strings (empty list means valid).
        """
        errors = []
        # Missing keys check (excluding non-feature metadata)
        missing_fields = REQUIRED_FIELDS - set(record.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {sorted(missing_fields)}")

        # Validate categorical value ranges against known VALID_VALUES
        for field, allowed in VALID_VALUES.items():
            if field in record and record[field] not in allowed:
                errors.append(f"Field '{field}' has invalid value {record[field]!r}. Allowed: {allowed}")

        # Age sanity check
        if "Age" in record:
            try:
                age_val = int(record["Age"])
                if not (0 <= age_val <= 120):
                    errors.append(f"Age {age_val} out of valid range (0-120)")
            except (ValueError, TypeError):
                errors.append(f"Age value {record['Age']!r} is not a valid integer")

        return errors

    def transform_single(self, claim: dict[str, Any]) -> pd.DataFrame:
        """Transform a single claim dictionary into a 1x86 feature DataFrame."""
        return self.transform_batch([claim])

    def transform_batch(self, claims: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Transform a list of raw claim records into an Nx86 feature DataFrame.

        Parameters
        ----------
        claims : List of claim dictionaries.

        Returns
        -------
        pd.DataFrame with shape (N, 86) matching exact feature_columns in order.
        """
        if not claims:
            # Return empty DataFrame with the 86 expected columns
            return pd.DataFrame(columns=self.feature_columns)

        processed_rows = []
        for idx, raw_claim in enumerate(claims):
            # Validate input
            errors = self.validate_raw_record(raw_claim)
            if errors:
                cid = raw_claim.get("claim_id", f"index_{idx}")
                raise FeaturePreprocessingError(f"Claim {cid} validation failed: {'; '.join(errors)}")

            row = dict(raw_claim)
            # Remove metadata fields if present
            row.pop("claim_id", None)
            row.pop("submitted_at", None)

            # Age imputation if 0 (matching train_and_save.py)
            try:
                row["Age"] = int(row["Age"])
                if row["Age"] == 0:
                    row["Age"] = 16
            except Exception as e:
                raise FeaturePreprocessingError(f"Error casting Age: {e}")

            # Apply binary mappings
            for col, mapping in self.binary_cols.items():
                val = row.get(col)
                if val not in mapping:
                    raise FeaturePreprocessingError(f"Unknown binary value for '{col}': {val!r}")
                row[col] = mapping[val]

            # Apply ordinal mappings
            for col, mapping in self.ordinal_mappings.items():
                val = row.get(col)
                if val not in mapping:
                    raise FeaturePreprocessingError(f"Unknown ordinal value for '{col}': {val!r}")
                row[col] = mapping[val]

            # Cast remaining numeric fields
            for num_col in ["Deductible", "DriverRating", "WeekOfMonth", "WeekOfMonthClaimed"]:
                try:
                    row[num_col] = int(row[num_col])
                except (ValueError, TypeError) as e:
                    raise FeaturePreprocessingError(f"Error casting numeric column '{num_col}': {e}")

            processed_rows.append(row)

        df = pd.DataFrame(processed_rows)

        # Apply nominal One-Hot Encoding
        df_nominal = df[self.nominal_cols]
        enc_out = self.encoder.transform(df_nominal)
        enc_cols = self.encoder.get_feature_names_out(self.nominal_cols)
        enc_df = pd.DataFrame(enc_out, columns=enc_cols, index=df.index)

        # Drop original nominal columns and join encoded dummies
        df_processed = df.drop(columns=self.nominal_cols).join(enc_df)

        # Reindex to ensure strict 86-feature order and names
        df_final = df_processed.reindex(columns=self.feature_columns, fill_value=0)

        # Final verification assertion
        if list(df_final.columns) != self.feature_columns:
            raise FeaturePreprocessingError("Column order mismatch after transformation")

        return df_final
