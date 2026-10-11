"""
ClaimShield AI — Phase 3: Kafka Producer
=========================================
Publishes insurance claim records as JSON messages to the
`insurance_claims` Kafka topic.

Usage
-----
# Send all 5 built-in synthetic sample claims (one-shot):
    python kafka_pipeline/kafka_producer.py

# Send claims continuously every N seconds (streaming simulation):
    python kafka_pipeline/kafka_producer.py --continuous --interval 3

# Send a specific number of random synthetic claims:
    python kafka_pipeline/kafka_producer.py --count 10

Run from the project root directory.

Notes
-----
* This producer does NOT interact with model.pkl / encoder.pkl / metadata.pkl.
* Messages contain raw (string) field values exactly as they would arrive from
  a claim intake system, BEFORE any ML pre-processing.
* Phase 4 will store these messages in HDFS.
* Phase 5 will add ML scoring inside the consumer pipeline.
"""

import argparse
import json
import logging
import random
import signal
import sys
import time
import uuid
import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow running from the project root as well as from kafka/
# ---------------------------------------------------------------------------
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kafka import KafkaProducer  # kafka-python-ng (installed package)
from kafka.errors import KafkaError, NoBrokersAvailable

from kafka_pipeline.config import KafkaConfig
from kafka_pipeline.sample_claims import SAMPLE_CLAIMS, VALID_VALUES, REQUIRED_FIELDS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("claimshield.producer")

# ---------------------------------------------------------------------------
# Graceful shutdown flag
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    log.info("Shutdown signal received (signal %s). Finishing current batch…", signum)
    _shutdown_requested = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_claim(claim: dict[str, Any]) -> list[str]:
    """
    Return a list of validation errors. An empty list means the claim is valid.

    Checks performed:
    1. All required fields are present.
    2. Categorical field values are within the allowed set.
    3. Age is a positive integer.
    """
    errors: list[str] = []

    # 1. Required field presence
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

    # 3. Age sanity check
    age = claim.get("Age")
    if age is not None:
        try:
            age_int = int(age)
            if not (16 <= age_int <= 120):
                errors.append(f"'Age' must be between 16 and 120, got {age_int}.")
        except (TypeError, ValueError):
            errors.append(f"'Age' must be an integer, got {age!r}.")

    return errors


# ---------------------------------------------------------------------------
# Random claim generator (for --count / --continuous modes)
# ---------------------------------------------------------------------------

def generate_random_claim() -> dict[str, Any]:
    """Build a fully-random synthetic claim drawn from the valid value sets."""
    return {
        "claim_id": f"CLM-{uuid.uuid4().hex[:10].upper()}",
        "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "Age": random.randint(16, 80),
        "Sex": random.choice(VALID_VALUES["Sex"]),
        "AccidentArea": random.choice(VALID_VALUES["AccidentArea"]),
        "Fault": random.choice(VALID_VALUES["Fault"]),
        "PoliceReportFiled": random.choice(VALID_VALUES["PoliceReportFiled"]),
        "WitnessPresent": random.choice(VALID_VALUES["WitnessPresent"]),
        "AgentType": random.choice(VALID_VALUES["AgentType"]),
        "VehiclePrice": random.choice(VALID_VALUES["VehiclePrice"]),
        "AgeOfVehicle": random.choice(VALID_VALUES["AgeOfVehicle"]),
        "AgeOfPolicyHolder": random.choice(VALID_VALUES["AgeOfPolicyHolder"]),
        "Days_Policy_Accident": random.choice(VALID_VALUES["Days_Policy_Accident"]),
        "Days_Policy_Claim": random.choice(VALID_VALUES["Days_Policy_Claim"]),
        "PastNumberOfClaims": random.choice(VALID_VALUES["PastNumberOfClaims"]),
        "NumberOfSuppliments": random.choice(VALID_VALUES["NumberOfSuppliments"]),
        "AddressChange_Claim": random.choice(VALID_VALUES["AddressChange_Claim"]),
        "NumberOfCars": random.choice(VALID_VALUES["NumberOfCars"]),
        "Make": random.choice(VALID_VALUES["Make"]),
        "PolicyType": random.choice(VALID_VALUES["PolicyType"]),
        "MaritalStatus": random.choice(VALID_VALUES["MaritalStatus"]),
        "VehicleCategory": random.choice(VALID_VALUES["VehicleCategory"]),
        "BasePolicy": random.choice(VALID_VALUES["BasePolicy"]),
        "Month": random.choice(VALID_VALUES["Month"]),
        "MonthClaimed": random.choice(VALID_VALUES["MonthClaimed"]),
        "DayOfWeek": random.choice(VALID_VALUES["DayOfWeek"]),
        "DayOfWeekClaimed": random.choice(VALID_VALUES["DayOfWeekClaimed"]),
        "Deductible": random.choice(VALID_VALUES["Deductible"]),
        "DriverRating": random.choice(VALID_VALUES["DriverRating"]),
        "WeekOfMonth": random.choice(VALID_VALUES["WeekOfMonth"]),
        "WeekOfMonthClaimed": random.choice(VALID_VALUES["WeekOfMonthClaimed"]),
    }


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

def build_producer(cfg: KafkaConfig) -> KafkaProducer:
    """
    Create and return a KafkaProducer.
    Raises NoBrokersAvailable if Kafka is not reachable.
    """
    log.info("Connecting to Kafka broker at %s …", cfg.bootstrap_servers)
    producer = KafkaProducer(
        bootstrap_servers=cfg.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks=cfg.producer_acks,
        retries=cfg.producer_retries,
        linger_ms=cfg.producer_linger_ms,
        # Connection timeout — fail fast instead of hanging indefinitely
        request_timeout_ms=15000,
        api_version_auto_timeout_ms=15000,
    )
    log.info("Producer connected. Bootstrap: %s", cfg.bootstrap_servers)
    return producer


def send_claim(
    producer: KafkaProducer,
    topic: str,
    claim: dict[str, Any],
) -> bool:
    """
    Validate and publish a single claim to Kafka.

    Returns True on success, False if validation failed.
    Raises KafkaError on broker-level failures (caller decides whether to abort).
    """
    claim_id = claim.get("claim_id", "<unknown>")

    # --- Validate before sending ---
    errors = validate_claim(claim)
    if errors:
        log.warning(
            "Claim %s REJECTED (validation failed). Errors:\n  %s",
            claim_id,
            "\n  ".join(errors),
        )
        return False

    # --- Publish ---
    future = producer.send(
        topic=topic,
        key=claim_id,          # use claim_id as partition key for ordering
        value=claim,
    )

    # Block until the broker acknowledges (with timeout)
    try:
        record_metadata = future.get(timeout=15)
        log.info(
            "✅ Published claim %s → topic=%s partition=%d offset=%d",
            claim_id,
            record_metadata.topic,
            record_metadata.partition,
            record_metadata.offset,
        )
        return True
    except KafkaError as exc:
        log.error("❌ Failed to publish claim %s: %s", claim_id, exc)
        raise


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ClaimShield AI — Kafka Producer (Phase 3)"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--continuous",
        action="store_true",
        help="Keep producing random claims until Ctrl-C.",
    )
    mode_group.add_argument(
        "--count",
        type=int,
        metavar="N",
        help="Send N randomly-generated synthetic claims then exit.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Seconds between messages in --continuous mode (default: 2).",
    )
    args = parser.parse_args()

    cfg = KafkaConfig()
    log.info("Configuration: %s", cfg)

    # --- Connect ---
    try:
        producer = build_producer(cfg)
    except NoBrokersAvailable:
        log.error(
            "Cannot reach Kafka at %s. "
            "Make sure Kafka is running (cd kafka && docker compose up -d).",
            cfg.bootstrap_servers,
        )
        sys.exit(1)
    except Exception as exc:
        log.error("Unexpected error while connecting to Kafka: %s", exc)
        sys.exit(1)

    topic = cfg.claims_topic
    sent = 0
    failed = 0

    try:
        if args.continuous:
            # ---- Streaming simulation mode ----
            log.info(
                "Continuous mode: publishing random claims every %.1fs. "
                "Press Ctrl-C to stop.",
                args.interval,
            )
            while not _shutdown_requested:
                claim = generate_random_claim()
                try:
                    ok = send_claim(producer, topic, claim)
                    if ok:
                        sent += 1
                    else:
                        failed += 1
                except KafkaError:
                    failed += 1
                time.sleep(args.interval)

        elif args.count:
            # ---- Fixed-count random mode ----
            log.info("Sending %d randomly-generated synthetic claims…", args.count)
            for i in range(args.count):
                if _shutdown_requested:
                    break
                claim = generate_random_claim()
                try:
                    ok = send_claim(producer, topic, claim)
                    if ok:
                        sent += 1
                    else:
                        failed += 1
                except KafkaError:
                    failed += 1

        else:
            # ---- Default: send the 5 pre-defined sample claims ----
            log.info(
                "Sending %d pre-defined synthetic sample claims…",
                len(SAMPLE_CLAIMS),
            )
            for claim in SAMPLE_CLAIMS:
                if _shutdown_requested:
                    break
                try:
                    ok = send_claim(producer, topic, claim)
                    if ok:
                        sent += 1
                    else:
                        failed += 1
                except KafkaError:
                    failed += 1

    finally:
        # Flush any buffered messages before closing
        log.info("Flushing producer buffer…")
        producer.flush(timeout=30)
        producer.close()
        log.info(
            "Producer closed. Summary: %d sent, %d failed.", sent, failed
        )


if __name__ == "__main__":
    main()
