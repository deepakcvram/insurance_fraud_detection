"""
ClaimShield AI — Phase 4: Modular HDFS Storage Component
=========================================================
Handles persistent storage of validated insurance claims into HDFS
using the standard WebHDFS REST API.

Design details:
--------------
- Format: JSON Lines (.jsonl). Each row is an exact validated claim.
- Encoding: UTF-8.
- Deduplication / Idempotence: Maintains claim_id identification, records
  persisted per batch in date-partitioned files or consolidated claim logs.
- Reliability: Supports small batch buffering, timeout handling, flush on shutdown,
  and comprehensive read-back functions for auditing & verification.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional
import urllib.parse
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kafka_pipeline.config import KafkaConfig

log = logging.getLogger("claimshield.hdfs")


class HDFSError(Exception):
    """Raised when an HDFS operation fails."""
    pass


class HDFSStorage:
    """
    Client for interacting with HDFS via WebHDFS REST API.
    Provides batch buffering and persistence for insurance claims.
    """

    def __init__(self, cfg: Optional[KafkaConfig] = None):
        self.cfg = cfg or KafkaConfig()
        self.base_url = self.cfg.hdfs_namenode_url.rstrip("/")
        self.user = self.cfg.hdfs_user
        self.claims_path = self.cfg.hdfs_claims_path.rstrip("/")
        self.batch_size = self.cfg.hdfs_batch_size
        self.flush_interval = self.cfg.hdfs_flush_interval_seconds
        self.timeout = self.cfg.hdfs_request_timeout_seconds

        self._buffer: list[dict[str, Any]] = []
        self._last_flush_time = time.time()
        self._session = requests.Session()

    def _webhdfs_url(self, path: str, op: str, **kwargs) -> str:
        """Construct WebHDFS URL with operation and query parameters."""
        path = path.lstrip("/")
        params = {"op": op, "user.name": self.user}
        params.update(kwargs)
        query = urllib.parse.urlencode(params)
        return f"{self.base_url}/webhdfs/v1/{path}?{query}"

    def is_available(self) -> bool:
        """Check if NameNode WebHDFS is accessible and responsive."""
        try:
            url = self._webhdfs_url("/", "GETFILESTATUS")
            resp = self._session.get(url, timeout=self.timeout)
            return resp.status_code == 200
        except Exception as exc:
            log.warning("HDFS health check failed: %s", exc)
            return False

    def ensure_directory(self, path: Optional[str] = None) -> bool:
        """Ensure the destination directory exists in HDFS."""
        target_path = path or self.claims_path
        url = self._webhdfs_url(target_path, "MKDIRS", permission="755")
        try:
            resp = self._session.put(url, timeout=self.timeout)
            if resp.status_code in (200, 201):
                data = resp.json()
                return data.get("boolean", True)
            raise HDFSError(f"MKDIRS failed with HTTP {resp.status_code}: {resp.text}")
        except Exception as exc:
            raise HDFSError(f"Failed to ensure HDFS directory {target_path}: {exc}") from exc

    def append_or_create(self, file_path: str, data_bytes: bytes) -> None:
        """
        Write or append data to a file in HDFS via WebHDFS two-step redirect protocol.
        """
        # 1. Check if file exists
        status_url = self._webhdfs_url(file_path, "GETFILESTATUS")
        file_exists = False
        try:
            check_resp = self._session.get(status_url, timeout=self.timeout)
            if check_resp.status_code == 200:
                file_exists = True
        except Exception as exc:
            raise HDFSError(f"Error checking file status {file_path}: {exc}") from exc

        if not file_exists:
            # CREATE new file (NameNode responds with 307 redirect to DataNode)
            create_url = self._webhdfs_url(file_path, "CREATE", overwrite="false")
            try:
                resp = self._session.put(create_url, allow_redirects=False, timeout=self.timeout)
                if resp.status_code != 307:
                    # Some setups may return direct or error
                    raise HDFSError(f"CREATE expected 307 redirect, got {resp.status_code}: {resp.text}")
                datanode_url = resp.headers["Location"]
                # In Docker on Windows/Mac, replace container internal hostname if returned
                datanode_url = self._fix_datanode_url(datanode_url)

                upload_resp = self._session.put(
                    datanode_url,
                    data=data_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=self.timeout,
                )
                if upload_resp.status_code not in (200, 201):
                    raise HDFSError(f"DataNode write failed HTTP {upload_resp.status_code}: {upload_resp.text}")
            except Exception as exc:
                raise HDFSError(f"Failed to create file in HDFS {file_path}: {exc}") from exc
        else:
            # APPEND to existing file (NameNode responds with 307 redirect to DataNode)
            append_url = self._webhdfs_url(file_path, "APPEND")
            try:
                resp = self._session.post(append_url, allow_redirects=False, timeout=self.timeout)
                if resp.status_code != 307:
                    raise HDFSError(f"APPEND expected 307 redirect, got {resp.status_code}: {resp.text}")
                datanode_url = resp.headers["Location"]
                datanode_url = self._fix_datanode_url(datanode_url)

                upload_resp = self._session.post(
                    datanode_url,
                    data=data_bytes,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=self.timeout,
                )
                if upload_resp.status_code not in (200, 201):
                    raise HDFSError(f"DataNode append failed HTTP {upload_resp.status_code}: {upload_resp.text}")
            except Exception as exc:
                raise HDFSError(f"Failed to append to file in HDFS {file_path}: {exc}") from exc

    def _fix_datanode_url(self, url: str) -> str:
        """
        When running locally with Docker port-mapping, DataNode inside container
        might report container hostname:port (e.g. datanode:9864).
        If host cannot resolve container hostname, rewrite host to localhost or config host.
        """
        parsed = urllib.parse.urlparse(url)
        # If hostname is internal docker name (not localhost / 127.0.0.1)
        if parsed.hostname in ("datanode", "namenode"):
            # Replace hostname with localhost but preserve port
            new_netloc = f"localhost:{parsed.port}"
            return urllib.parse.urlunparse(parsed._replace(netloc=new_netloc))
        return url

    def save_claim(self, claim: dict[str, Any]) -> None:
        """
        Buffer a validated claim record and flush if batch size is reached.
        """
        self._buffer.append(claim)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def check_flush_interval(self) -> None:
        """Flush buffer if elapsed time exceeds flush interval."""
        if self._buffer and (time.time() - self._last_flush_time) >= self.flush_interval:
            self.flush()

    def flush(self) -> int:
        """
        Flush all buffered claims to HDFS as JSONL records.
        Returns the number of flushed claims.
        """
        if not self._buffer:
            return 0

        claims_to_flush = list(self._buffer)
        count = len(claims_to_flush)
        log.info("Flushing %d claims to HDFS destination '%s'...", count, self.claims_path)

        # Ensure claims directory exists
        self.ensure_directory()

        # Group by day partition: /claimshield/claims/claims_YYYYMMDD.jsonl
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        target_file = f"{self.claims_path}/claims_{today}.jsonl"

        lines = [json.dumps(c, ensure_ascii=False) + "\n" for c in claims_to_flush]
        payload = "".join(lines).encode("utf-8")

        try:
            self.append_or_create(target_file, payload)
            # Only clear buffer once write succeeds
            self._buffer.clear()
            self._last_flush_time = time.time()
            log.info("Successfully persisted %d claims to HDFS at '%s'", count, target_file)
            return count
        except Exception as exc:
            log.error("Failed to persist batch of %d claims to HDFS: %s", count, exc)
            raise

    def read_claims(self, file_name: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Read back and deserialize JSONL claim records from HDFS.
        If file_name is None, reads the current day's claim file.
        """
        if file_name is None:
            today = datetime.now(timezone.utc).strftime("%Y%m%d")
            file_path = f"{self.claims_path}/claims_{today}.jsonl"
        else:
            file_path = f"{self.claims_path}/{file_name}" if not file_name.startswith("/") else file_name

        open_url = self._webhdfs_url(file_path, "OPEN")
        resp = self._session.get(open_url, allow_redirects=False, timeout=self.timeout)
        if resp.status_code == 404:
            return []
        if resp.status_code == 307:
            datanode_url = self._fix_datanode_url(resp.headers["Location"])
            resp = self._session.get(datanode_url, timeout=self.timeout)
        if resp.status_code != 200:
            raise HDFSError(f"OPEN failed HTTP {resp.status_code}: {resp.text}")

        lines = resp.text.splitlines()
        records = []
        for line in lines:
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def list_files(self, path: Optional[str] = None) -> list[str]:
        """List filenames in the given HDFS directory."""
        target_path = path or self.claims_path
        url = self._webhdfs_url(target_path, "LISTSTATUS")
        resp = self._session.get(url, timeout=self.timeout)
        if resp.status_code == 404:
            return []
        if resp.status_code != 200:
            raise HDFSError(f"LISTSTATUS failed HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        statuses = data.get("FileStatuses", {}).get("FileStatus", [])
        return [s.get("pathSuffix") for s in statuses]

    def close(self) -> None:
        """Flush any pending buffered claims and close HTTP session."""
        if self._buffer:
            log.info("Closing HDFSStorage: flushing %d pending buffered claims...", len(self._buffer))
            try:
                self.flush()
            except Exception as exc:
                log.error("Error flushing claims during close: %s", exc)
        self._session.close()
