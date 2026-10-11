"""
ClaimShield AI — Pipeline Configuration (Phases 3, 4 & 5)
==========================================================
Central place for all Kafka, HDFS & Phase 5 Analytics settings.
Override any value via environment variables so secrets
never need to be hardcoded.

Usage:
    from kafka_pipeline.config import KafkaConfig
    cfg = KafkaConfig()
    print(cfg.bootstrap_servers)
    print(cfg.hdfs_namenode_url)
    print(cfg.fraud_threshold)
"""

import os


class KafkaConfig:
    """Pipeline connection, topic, HDFS storage, and Phase 5 analytics settings."""

    # ------------------------------------------------------------------ #
    # Kafka Broker                                                         #
    # ------------------------------------------------------------------ #
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    # ------------------------------------------------------------------ #
    # Topics                                                               #
    # ------------------------------------------------------------------ #
    claims_topic: str = os.getenv("KAFKA_CLAIMS_TOPIC", "insurance_claims")

    # ------------------------------------------------------------------ #
    # Producer                                                             #
    # ------------------------------------------------------------------ #
    producer_acks: str = os.getenv("KAFKA_PRODUCER_ACKS", "all")   # strongest durability
    producer_retries: int = int(os.getenv("KAFKA_PRODUCER_RETRIES", "3"))
    producer_linger_ms: int = int(os.getenv("KAFKA_PRODUCER_LINGER_MS", "10"))  # small batching

    # ------------------------------------------------------------------ #
    # Consumer                                                             #
    # ------------------------------------------------------------------ #
    consumer_group_id: str = os.getenv("KAFKA_CONSUMER_GROUP_ID", "claimshield-fraud-detector")
    consumer_auto_offset_reset: str = os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest")
    # Manual offset commit by default to ensure no claim is lost
    consumer_enable_auto_commit: bool = (
        os.getenv("KAFKA_ENABLE_AUTO_COMMIT", "false").lower() == "true"
    )
    consumer_session_timeout_ms: int = int(os.getenv("KAFKA_SESSION_TIMEOUT_MS", "30000"))
    consumer_max_poll_interval_ms: int = int(os.getenv("KAFKA_MAX_POLL_INTERVAL_MS", "300000"))

    # ------------------------------------------------------------------ #
    # Phase 4 HDFS Storage Settings                                        #
    # ------------------------------------------------------------------ #
    # WebHDFS REST API endpoint on NameNode
    hdfs_namenode_url: str = os.getenv("HDFS_NAMENODE_URL", "http://localhost:9870")
    hdfs_user: str = os.getenv("HDFS_USER", "root")
    hdfs_claims_path: str = os.getenv("HDFS_CLAIMS_PATH", "/claimshield/claims")
    hdfs_batch_size: int = int(os.getenv("HDFS_BATCH_SIZE", "5"))
    hdfs_flush_interval_seconds: float = float(os.getenv("HDFS_FLUSH_INTERVAL", "5.0"))
    hdfs_request_timeout_seconds: int = int(os.getenv("HDFS_TIMEOUT", "10"))

    # ------------------------------------------------------------------ #
    # Phase 5 Historical Analytics & Batch Inference Settings              #
    # ------------------------------------------------------------------ #
    fraud_threshold: float = float(os.getenv("FRAUD_THRESHOLD", "0.70"))
    model_path: str = os.getenv("MODEL_PATH", "model.pkl")
    encoder_path: str = os.getenv("ENCODER_PATH", "encoder.pkl")
    metadata_path: str = os.getenv("METADATA_PATH", "metadata.pkl")
    analytics_output_dir: str = os.getenv("ANALYTICS_OUTPUT_DIR", "analytics_output")
    hdfs_analytics_path: str = os.getenv("HDFS_ANALYTICS_PATH", "/claimshield/analytics")

    def __repr__(self) -> str:
        return (
            f"KafkaConfig("
            f"bootstrap={self.bootstrap_servers!r}, "
            f"topic={self.claims_topic!r}, "
            f"group={self.consumer_group_id!r}, "
            f"hdfs={self.hdfs_namenode_url!r}, "
            f"hdfs_path={self.hdfs_claims_path!r}, "
            f"threshold={self.fraud_threshold})"
        )
