# 🛡️ ClaimShield AI — Vehicle Insurance Claim Fraud Detection

> An end-to-end machine learning platform for real-time and historical insurance claim fraud detection, built across five incremental engineering phases.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-red?logo=streamlit)](https://streamlit.io/)
[![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-3.9.0-black?logo=apachekafka)](https://kafka.apache.org/)
[![Hadoop](https://img.shields.io/badge/Hadoop%20HDFS-3.3.6-yellow?logo=apachehadoop)](https://hadoop.apache.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://www.docker.com/)

---

## 📌 Project Overview

ClaimShield AI detects potentially fraudulent vehicle insurance claims using a trained XGBoost model. It supports both interactive single-claim assessment via a Streamlit dashboard and continuous real-time processing via a Kafka streaming pipeline. Validated claims are persisted in HDFS for historical batch analysis.

**Key capabilities:**

| Capability | Technology |
|---|---|
| ML-based fraud scoring | XGBoost + scikit-learn preprocessing |
| Interactive claim assessment | Streamlit dashboard |
| Real-time claim streaming | Apache Kafka (KRaft mode) |
| Persistent claim storage | Hadoop HDFS (WebHDFS REST) |
| Historical batch inference | Python + Pandas |
| Containerized infrastructure | Docker Compose |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ClaimShield AI                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Streamlit Dashboard]  ──────────────────────────────────► │
│       (app.py)            Real-time single-claim scoring    │
│                                                             │
│  [Kafka Producer]  ──►  [Kafka Topic: insurance_claims]     │
│  (kafka_producer.py)          │                             │
│                               ▼                             │
│                     [Kafka Consumer]                        │
│                     (kafka_consumer.py)                     │
│                               │                             │
│                    JSON deserialization                      │
│                    Schema validation                         │
│                               │                             │
│                               ▼                             │
│                     [HDFS Storage]  ◄── WebHDFS REST        │
│                     (hdfs_storage.py)                       │
│                               │                             │
│                               ▼                             │
│                  [Historical Analytics]                      │
│                 (historical_analytics.py)                   │
│                               │                             │
│           ┌───────────────────┼───────────────────┐         │
│           ▼                   ▼                   ▼         │
│  Batch Inference       Analytics Report    Investigation     │
│  Results CSV          JSON Summary         Queue CSV         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
Vehicle-Insurance-Claim-Fraud-Detection-main/
│
├── app.py                        # Streamlit interactive dashboard
├── train_and_save.py             # ML model training script
├── model.ipynb                   # Exploratory notebook
├── dataset.csv                   # Training dataset (15,419 records)
│
├── model.pkl                     # Trained XGBClassifier
├── encoder.pkl                   # Fitted OneHotEncoder (nominal columns)
├── metadata.pkl                  # Binary/ordinal mappings + feature column list
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
│
└── kafka_pipeline/
    ├── docker-compose.yml        # Kafka + HDFS NameNode + DataNode
    ├── config.py                 # Centralised configuration (KafkaConfig)
    ├── kafka_producer.py         # Publishes claim events to Kafka
    ├── kafka_consumer.py         # Consumes, validates, and stores claims
    ├── sample_claims.py          # 5 predefined + random claim generators
    ├── hdfs_storage.py           # WebHDFS REST client (HDFSStorage class)
    ├── feature_preprocessor.py   # Shared preprocessing (FeaturePreprocessor)
    ├── historical_analytics.py   # Batch inference pipeline
    ├── test_phase5.py            # Automated test suite (10 tests)
    └── hadoop_config/
        ├── core-site.xml         # HDFS core configuration
        └── hdfs-site.xml         # HDFS replication configuration
```

---

## 🚀 Quickstart

### Prerequisites

- Python 3.10+ (tested on 3.13)
- Docker Desktop (for Kafka + HDFS)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Vehicle-Insurance-Claim-Fraud-Detection.git
cd Vehicle-Insurance-Claim-Fraud-Detection
```

### 2. Create a Virtual Environment and Install Dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Train the Model (if artifacts are not present)

```bash
python train_and_save.py
```

This produces `model.pkl`, `encoder.pkl`, and `metadata.pkl` in the project root.

### 4. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to assess individual claims interactively.

---

## ⚙️ Kafka + HDFS Pipeline

### Start Infrastructure

```bash
cd kafka_pipeline
docker compose up -d
```

This starts three containers:

| Container | Role | Port |
|---|---|---|
| `claimshield-kafka` | Kafka broker (KRaft mode) | 9092 |
| `claimshield-namenode` | HDFS NameNode | 9870 (WebHDFS), 9000 (IPC) |
| `claimshield-datanode` | HDFS DataNode | 9864 |

Wait ~30 seconds for all containers to become healthy:

```bash
docker compose ps
```

### Start the Consumer

Open a terminal in the project root:

```bash
.venv\Scripts\python kafka_pipeline/kafka_consumer.py
```

The consumer will listen on the `insurance_claims` topic and persist validated claims to HDFS.

### Publish Sample Claims

Open a **second** terminal in the project root:

```bash
.venv\Scripts\python kafka_pipeline/kafka_producer.py
```

This publishes the 5 predefined sample claims. The consumer terminal will display validation results for each.

### Run Historical Batch Analysis

After claims have been stored in HDFS, run the batch inference pipeline:

```bash
.venv\Scripts\python kafka_pipeline/historical_analytics.py
```

Output files are written to `analytics_output/`:

| File | Contents |
|---|---|
| `batch_inference_results_<date>.csv` | Per-claim fraud scores and decisions |
| `historical_analytics_report_<date>.json` | Aggregate statistics and summary |
| `investigation_queue.csv` | Claims flagged for manual review |
| `rejected_claims_<date>.json` | Records that failed validation |

### Run Automated Tests

```bash
.venv\Scripts\python -m pytest kafka_pipeline/test_phase5.py -v
```

---

## 🧠 ML Model Details

| Attribute | Value |
|---|---|
| Algorithm | XGBoost (`XGBClassifier`) |
| Training records | 12,335 |
| Input features | 86 (after encoding) |
| Class imbalance handling | `scale_pos_weight` |
| Default fraud threshold | 0.15 (dashboard) / 0.70 (batch pipeline) |

### Feature Pipeline

Raw claim fields (29 domain columns) → **FeaturePreprocessor** → 86 model features

| Transformation | Columns |
|---|---|
| **Binary encoding** | `Sex`, `AccidentArea`, `Fault`, `PoliceReportFiled`, `WitnessPresent`, `AgentType` |
| **Ordinal encoding** | `VehiclePrice`, `AgeOfVehicle`, `AgeOfPolicyHolder`, `Days_Policy_Accident`, `Days_Policy_Claim`, `PastNumberOfClaims`, `NumberOfSuppliments`, `AddressChange_Claim`, `NumberOfCars` |
| **One-hot encoding** | `Make`, `PolicyType`, `MaritalStatus`, `VehicleCategory`, `BasePolicy`, `Month`, `MonthClaimed`, `DayOfWeek`, `DayOfWeekClaimed` |
| **Numeric (passthrough)** | `Age`, `Deductible`, `DriverRating`, `WeekOfMonth`, `WeekOfMonthClaimed` |

---

## 📋 Dataset

The model is trained on the [Vehicle Insurance Claim Fraud Detection dataset](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) from Kaggle.

- **Total records:** 15,419
- **Fraud prevalence:** ~6% (highly imbalanced)
- **Original columns:** 33

---

## ⚠️ Known Limitations

1. **Pickle version compatibility:** `model.pkl` and `encoder.pkl` were serialized with specific library versions. If you see `UserWarning: Trying to unpickle estimator from version X`, re-run `train_and_save.py` in your environment to regenerate the artifacts.

2. **HDFS requires Docker:** The Kafka consumer writes to HDFS via WebHDFS REST. Without Docker running, the consumer still validates claims but HDFS writes will fail gracefully with a warning.

3. **Windows hostname resolution:** HDFS DataNode redirects use Docker-internal hostnames. The `hdfs_storage.py` client handles this by rewriting redirect URLs to `localhost` automatically.

4. **Unknown vehicle makes:** The OneHotEncoder is fitted on training data. If a claim contains a `Make` value not seen during training, it is encoded as all-zeros (safe — `handle_unknown='ignore'`).

---

## 🗂️ Development Phases

| Phase | Description | Key Files |
|---|---|---|
| **1** | XGBoost model + Streamlit dashboard | `train_and_save.py`, `app.py` |
| **2** | Exploratory data analysis | `model.ipynb` |
| **3** | Kafka streaming pipeline | `kafka_producer.py`, `kafka_consumer.py`, `docker-compose.yml` |
| **4** | HDFS persistence | `hdfs_storage.py`, `docker-compose.yml` (extended) |
| **5** | Historical batch inference | `feature_preprocessor.py`, `historical_analytics.py`, `test_phase5.py` |

---

## 🛠️ Tech Stack

- **Python 3.13** — core language
- **XGBoost + scikit-learn** — machine learning
- **Streamlit** — interactive dashboard
- **Apache Kafka 3.9.0** — real-time event streaming (KRaft mode, no ZooKeeper)
- **Hadoop HDFS 3.3.6** — distributed persistent storage
- **Docker Compose** — container orchestration
- **Pandas + NumPy** — data manipulation and analytics
- **kafka-python-ng** — Python Kafka client

---

## 📄 License

This project is intended for educational and portfolio purposes.
