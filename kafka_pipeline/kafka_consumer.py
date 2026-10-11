"""
ClaimShield AI — Phase 3 & 4: Kafka Consumer with HDFS Integration
===================================================================
Subscribes to the `insurance_claims` topic, deserialises JSON messages,
validates them, and persists valid claims to Apache Hadoop HDFS.

Reliability & Delivery Guarantees:
----------------------------------
* At-Least-Once Delivery: Offsets are committed ONLY after valid claims are
  successfully buffered and committed to HDFS.
* Failure Strategy: If HDFS is temporarily unavailable, claims are NOT silently
  lost. The consumer pauses polling and retries connecting to HDFS with
  exponential backoff until HDFS recovers. Offsets remain uncommitted until
  persistence succeeds.
* Malformed Message Handling: Bad JSON or invalid schemas are logged and skipped
  from entering the validated HDFS dataset.
* Graceful Shutdown: SIGINT (Ctrl-C) or SIGTERM immediately flushes all pending
  buffered records to HDFS before closing the consumer and committing offsets.

Usage:
------
    python kafka_pipeline/kafka_consumer.py
    python kafka_pipeline/kafka_consumer.py --from-beginning
"""

import argparse
import json
import logging
import signal
import sys
import time
from typing import Any, Optional
import requests

import os
import selectors

# Workaround for Python 3.13 on Windows where closed sockets raise ValueError in SelectSelector.unregister
_orig_unregister = selectors.SelectSelector.unregister
def _safe_unregister(self, fileobj):
    try:
        return _orig_unregister(self, fileobj)
    except (ValueError, KeyError, OSError):
        pass
selectors.SelectSelector.unregister = _safe_unregister

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kafka import KafkaConsumer, TopicPartition, OffsetAndMetadata
from kafka.errors import KafkaError, NoBrokersAvailable

from kafka_pipeline.config import KafkaConfig
from kafka_pipeline.sample_claims import REQUIRED_FIELDS, VALID_VALUES
from kafka_pipeline.hdfs_storage import HDFSStorage, HDFSError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("claimshield.consumer")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    log.info("Shutdown signal received (signal %s). Stopping consumer gracefully…", signum)
    _shutdown_requested = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_claim(claim: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty list = valid)."""
    errors: list[str] = []

    # 1. Required fields
    missing = REQUIRED_FIELDS - set(claim.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")

    # 2. Categorical value ranges
    for field, allowed in VALID_VALUES.items():
        if field in claim and claim[field] not in allowed:
            errors.append(
                f"Field '{field}' has invalid value {claim[field]!r}. "
                f"Allowed: {allowed}"
            )

    # 3. Age range
    age = claim.get("Age")
    if age is not None:
        try:
            age_int = int(age)
            if not (16 <= age_int <= 120):
                errors.append(f"'Age' out of range: {age_int}")
        except (TypeError, ValueError):
            errors.append(f"'Age' must be an integer, got {age!r}.")

    return errors


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process_claim(
    claim: dict[str, Any],
    partition: int,
    offset: int,
    hdfs_client: Optional[HDFSStorage] = None,
) -> None:
    """
    Handle a single validated claim record.

    Parameters
    ----------
    claim       : Deserialised claim dictionary.
    partition   : Kafka partition the message arrived on.
    offset      : Message offset within the partition.
    hdfs_client : HDFS storage client instance.
    """
    claim_id = claim.get("claim_id", "<no-id>")
    submitted_at = claim.get("submitted_at", "<no-timestamp>")
    age = claim.get("Age")
    vehicle_make = claim.get("Make")
    vehicle_price = claim.get("VehiclePrice")
    fault = claim.get("Fault")
    police_filed = claim.get("PoliceReportFiled")
    witness = claim.get("WitnessPresent")
    past_claims = claim.get("PastNumberOfClaims")
    supplements = claim.get("NumberOfSuppliments")

    log.info(
        "📥 Received claim | ID=%s | submitted=%s | "
        "age=%s | make=%s | price=%s | fault=%s | "
        "police=%s | witness=%s | past_claims=%s | supplements=%s | "
        "[partition=%d offset=%d]",
        claim_id, submitted_at,
        age, vehicle_make, vehicle_price, fault,
        police_filed, witness, past_claims, supplements,
        partition, offset,
    )

    # ------------------------------------------------------------------
    # PHASE_4_HOOK: HDFS Storage
    # ------------------------------------------------------------------
    if hdfs_client:
        hdfs_client.save_claim(claim)
        log.info("💾 Claim %s queued/buffered for HDFS persistence", claim_id)


# ---------------------------------------------------------------------------
# Consumer builder
# ---------------------------------------------------------------------------

def build_consumer(cfg: KafkaConfig, from_beginning: bool) -> KafkaConsumer:
    """Create and return a KafkaConsumer."""
    auto_offset_reset = "earliest" if from_beginning else cfg.consumer_auto_offset_reset

    log.info(
        "Connecting to Kafka at %s | group=%s | topic=%s | offset=%s",
        cfg.bootstrap_servers,
        cfg.consumer_group_id,
        cfg.claims_topic,
        auto_offset_reset,
    )

    consumer = KafkaConsumer(
        cfg.claims_topic,
        bootstrap_servers=cfg.bootstrap_servers,
        group_id=cfg.consumer_group_id,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=cfg.consumer_enable_auto_commit,
        session_timeout_ms=cfg.consumer_session_timeout_ms,
        max_poll_interval_ms=cfg.consumer_max_poll_interval_ms,
        value_deserializer=lambda raw: _safe_deserialize(raw),
        consumer_timeout_ms=1000,
    )

    log.info("Consumer subscribed to topic '%s'.", cfg.claims_topic)
    return consumer


def _safe_deserialize(raw: bytes) -> Any:
    """Deserialise raw bytes as JSON safely."""
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.warning("Could not deserialise message bytes: %s", exc)
        return raw


# ---------------------------------------------------------------------------
# Main consumer loop
# ---------------------------------------------------------------------------

def run_consumer(cfg: KafkaConfig, from_beginning: bool) -> None:
    """Poll Kafka and process claim messages until shutdown is requested."""

    # Initialize HDFS storage component
    hdfs_client = HDFSStorage(cfg)
    log.info("HDFS storage client initialized for %s", cfg.hdfs_namenode_url)

    # Connect to Kafka broker
    max_connect_attempts = 5
    for attempt in range(1, max_connect_attempts + 1):
        try:
            consumer = build_consumer(cfg, from_beginning)
            break
        except NoBrokersAvailable:
            if attempt == max_connect_attempts:
                log.error(
                    "Cannot reach Kafka at %s after %d attempts. "
                    "Make sure Kafka is running (docker compose up -d).",
                    cfg.bootstrap_servers,
                    max_connect_attempts,
                )
                sys.exit(1)
            log.warning(
                "Broker unavailable (attempt %d/%d). Retrying in 5 s…",
                attempt, max_connect_attempts,
            )
            time.sleep(5)

    received = 0
    valid = 0
    invalid = 0
    malformed = 0

    log.info("Consumer loop started. Press Ctrl-C to stop.")

    try:
        while not _shutdown_requested:
            try:
                # poll() explicitly with 1000ms timeout
                msg_pack = consumer.poll(timeout_ms=1000)
                if _shutdown_requested:
                    break

                for topic_partition, messages in msg_pack.items():
                    for message in messages:
                        if _shutdown_requested:
                            break

                    received += 1
                    claim = message.value

                    # ---- Detect deserialisation failures ----
                    if isinstance(claim, (bytes, bytearray)):
                        malformed += 1
                        log.error(
                            "❌ Malformed message [partition=%d offset=%d]: "
                            "could not parse JSON. Raw (first 200 bytes): %r",
                            message.partition,
                            message.offset,
                            claim[:200],
                        )
                        # We commit offset for unparseable poison messages so pipeline isn't blocked
                        _commit_offset_safely(consumer, message)
                        continue

                    if not isinstance(claim, dict):
                        malformed += 1
                        log.error(
                            "❌ Unexpected message type %s [partition=%d offset=%d]. Skipping.",
                            type(claim).__name__,
                            message.partition,
                            message.offset,
                        )
                        _commit_offset_safely(consumer, message)
                        continue

                    # ---- Schema validation ----
                    errors = validate_claim(claim)
                    if errors:
                        invalid += 1
                        log.warning(
                            "⚠️  Invalid claim [partition=%d offset=%d]: %s",
                            message.partition,
                            message.offset,
                            "; ".join(errors),
                        )
                        _commit_offset_safely(consumer, message)
                        continue

                    # ---- Process valid claim with HDFS retry protection ----
                    valid += 1
                    persisted = False
                    while not persisted and not _shutdown_requested:
                        try:
                            process_claim(claim, message.partition, message.offset, hdfs_client)
                            # Explicitly flush to guarantee immediate HDFS persistence & offset commit
                            hdfs_client.flush()
                            _commit_offset_safely(consumer, message)
                            persisted = True
                        except (HDFSError, requests.RequestException) as hdfs_err:
                            log.error(
                                "🚨 HDFS persistence failed for claim %s: %s. "
                                "Retrying in 5 seconds without committing Kafka offset...",
                                claim.get("claim_id", "<unknown>"),
                                hdfs_err,
                            )
                            time.sleep(5.0)

                # Periodic flush check for any time-based buffered items
                try:
                    hdfs_client.check_flush_interval()
                except Exception as exc:
                    log.warning("Periodic HDFS flush check failed: %s", exc)

            except KafkaError as exc:
                log.error("Kafka error during poll: %s. Will retry…", exc)
                time.sleep(3)

    finally:
        log.info("Shutting down consumer and flushing pending records to HDFS…")
        try:
            hdfs_client.close()
        except Exception as exc:
            log.warning("Error while closing HDFS client: %s", exc)

        try:
            consumer.close()
        except Exception as exc:
            log.warning("Error while closing Kafka consumer: %s", exc)

        log.info(
            "Consumer stopped. Summary: %d received | %d valid | "
            "%d invalid schema | %d malformed JSON.",
            received, valid, invalid, malformed,
        )


def _commit_offset_safely(consumer: KafkaConsumer, message) -> None:
    """Manually commit offset for the processed message."""
    try:
        tp = TopicPartition(message.topic, message.partition)
        # Commit next offset (+1)
        consumer.commit({tp: OffsetAndMetadata(message.offset + 1, None)})
    except Exception as exc:
        log.debug("Offset commit notification: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ClaimShield AI — Kafka Consumer with HDFS Integration (Phase 4)"
    )
    parser.add_argument(
        "--from-beginning",
        action="store_true",
        help="Consume all messages from the beginning of the topic "
             "(overrides committed offsets for this run).",
    )
    args = parser.parse_args()

    cfg = KafkaConfig()
    log.info("Configuration: %s", cfg)

    run_consumer(cfg, from_beginning=args.from_beginning)


if __name__ == "__main__":
    main()
