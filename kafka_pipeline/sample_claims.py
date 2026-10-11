"""
ClaimShield AI — Phase 3: Synthetic Sample Claims
==================================================
These records use the exact field names required by the ClaimShield AI
feature schema (see train_and_save.py and app.py).

IMPORTANT: These are SYNTHETIC records generated for pipeline testing only.
They do NOT represent real insurance claims or real people.
The consumer intentionally does NOT run ML inference on raw Kafka messages
(Phase 3 scope). Full ML scoring will be wired in after Phase 4 / Phase 5.

Field reference
---------------
Binary (string values that get mapped to 0/1 later):
  Sex, AccidentArea, Fault, PoliceReportFiled, WitnessPresent, AgentType

Ordinal (categorical strings with an ordered rank):
  VehiclePrice, AgeOfVehicle, AgeOfPolicyHolder,
  Days_Policy_Accident, Days_Policy_Claim,
  PastNumberOfClaims, NumberOfSuppliments, AddressChange_Claim,
  NumberOfCars

Nominal (one-hot encoded):
  Make, PolicyType, MaritalStatus, VehicleCategory, BasePolicy,
  Month, MonthClaimed, DayOfWeek, DayOfWeekClaimed

Numeric:
  Age, Deductible, DriverRating, WeekOfMonth, WeekOfMonthClaimed

Kafka-only metadata (NOT model features):
  claim_id, submitted_at
"""

import uuid
import datetime

# ---------------------------------------------------------------------------
# Valid categorical values (kept in sync with train_and_save.py mappings)
# ---------------------------------------------------------------------------
VALID_VALUES = {
    "Sex": ["Male", "Female"],
    "AccidentArea": ["Urban", "Rural"],
    "Fault": ["Policy Holder", "Third Party"],
    "PoliceReportFiled": ["Yes", "No"],
    "WitnessPresent": ["Yes", "No"],
    "AgentType": ["External", "Internal"],
    "VehiclePrice": [
        "less than 20000", "20000 to 29000", "30000 to 39000",
        "40000 to 59000", "60000 to 69000", "more than 69000",
    ],
    "AgeOfVehicle": [
        "new", "2 years", "3 years", "4 years",
        "5 years", "6 years", "7 years", "more than 7",
    ],
    "AgeOfPolicyHolder": [
        "16 to 17", "18 to 20", "21 to 25", "26 to 30",
        "31 to 35", "36 to 40", "41 to 50", "51 to 65", "over 65",
    ],
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "AddressChange_Claim": [
        "no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years",
    ],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
    "Make": [
        "Pontiac", "Toyota", "Honda", "Mazda", "Chevrolet", "Accura", "Ford",
        "VW", "Dodge", "Saab", "Mercury", "Saturn", "Nisson", "BMW",
        "Jaguar", "Porche", "Mecedes", "Ferrari", "Lexus",
    ],
    "PolicyType": [
        "Sedan - Collision", "Sedan - Liability", "Sedan - All Perils",
        "Sport - Collision", "Utility - All Perils", "Utility - Collision",
        "Sport - All Perils", "Utility - Liability", "Sport - Liability",
    ],
    "MaritalStatus": ["Married", "Single", "Divorced", "Widow"],
    "VehicleCategory": ["Sedan", "Sport", "Utility"],
    "BasePolicy": ["Collision", "Liability", "All Perils"],
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "MonthClaimed": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "DayOfWeek": [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    ],
    "DayOfWeekClaimed": [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    ],
    "Deductible": [300, 400, 500, 700],
    "DriverRating": [1, 2, 3, 4],
    "WeekOfMonth": [1, 2, 3, 4, 5],
    "WeekOfMonthClaimed": [1, 2, 3, 4, 5],
}

# ---------------------------------------------------------------------------
# Required fields for a valid claim message
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = {
    # Kafka metadata
    "claim_id", "submitted_at",
    # Binary
    "Sex", "AccidentArea", "Fault", "PoliceReportFiled", "WitnessPresent", "AgentType",
    # Ordinal
    "VehiclePrice", "AgeOfVehicle", "AgeOfPolicyHolder",
    "Days_Policy_Accident", "Days_Policy_Claim",
    "PastNumberOfClaims", "NumberOfSuppliments", "AddressChange_Claim", "NumberOfCars",
    # Nominal
    "Make", "PolicyType", "MaritalStatus", "VehicleCategory", "BasePolicy",
    "Month", "MonthClaimed", "DayOfWeek", "DayOfWeekClaimed",
    # Numeric
    "Age", "Deductible", "DriverRating", "WeekOfMonth", "WeekOfMonthClaimed",
}


def _make_id() -> str:
    """Generate a unique claim ID prefixed with CLM-."""
    return f"CLM-{uuid.uuid4().hex[:10].upper()}"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Pre-defined synthetic test claims
# ---------------------------------------------------------------------------
# Each claim is a complete, self-contained dictionary.
# Entries are varied to cover different risk profiles.

SAMPLE_CLAIMS: list[dict] = [
    # ---- Claim 1: Low-risk profile ----------------------------------------
    {
        "claim_id": _make_id(),
        "submitted_at": _now_iso(),
        "Age": 42,
        "Sex": "Male",
        "AccidentArea": "Urban",
        "Fault": "Third Party",
        "PoliceReportFiled": "Yes",
        "WitnessPresent": "Yes",
        "AgentType": "External",
        "VehiclePrice": "20000 to 29000",
        "AgeOfVehicle": "3 years",
        "AgeOfPolicyHolder": "41 to 50",
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "PastNumberOfClaims": "none",
        "NumberOfSuppliments": "none",
        "AddressChange_Claim": "no change",
        "NumberOfCars": "2 vehicles",
        "Make": "Toyota",
        "PolicyType": "Sedan - Collision",
        "MaritalStatus": "Married",
        "VehicleCategory": "Sedan",
        "BasePolicy": "Collision",
        "Month": "Mar",
        "MonthClaimed": "Mar",
        "DayOfWeek": "Monday",
        "DayOfWeekClaimed": "Wednesday",
        "Deductible": 400,
        "DriverRating": 2,
        "WeekOfMonth": 2,
        "WeekOfMonthClaimed": 2,
    },
    # ---- Claim 2: High-risk profile (multiple fraud indicators) -----------
    {
        "claim_id": _make_id(),
        "submitted_at": _now_iso(),
        "Age": 23,
        "Sex": "Male",
        "AccidentArea": "Urban",
        "Fault": "Policy Holder",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "AgentType": "External",
        "VehiclePrice": "more than 69000",
        "AgeOfVehicle": "more than 7",
        "AgeOfPolicyHolder": "21 to 25",
        "Days_Policy_Accident": "1 to 7",
        "Days_Policy_Claim": "8 to 15",
        "PastNumberOfClaims": "more than 4",
        "NumberOfSuppliments": "more than 5",
        "AddressChange_Claim": "under 6 months",
        "NumberOfCars": "1 vehicle",
        "Make": "Ferrari",
        "PolicyType": "Sport - All Perils",
        "MaritalStatus": "Single",
        "VehicleCategory": "Sport",
        "BasePolicy": "All Perils",
        "Month": "Jan",
        "MonthClaimed": "Jan",
        "DayOfWeek": "Friday",
        "DayOfWeekClaimed": "Monday",
        "Deductible": 300,
        "DriverRating": 4,
        "WeekOfMonth": 1,
        "WeekOfMonthClaimed": 1,
    },
    # ---- Claim 3: Medium-risk profile ------------------------------------
    {
        "claim_id": _make_id(),
        "submitted_at": _now_iso(),
        "Age": 36,
        "Sex": "Female",
        "AccidentArea": "Rural",
        "Fault": "Policy Holder",
        "PoliceReportFiled": "Yes",
        "WitnessPresent": "No",
        "AgentType": "Internal",
        "VehiclePrice": "40000 to 59000",
        "AgeOfVehicle": "5 years",
        "AgeOfPolicyHolder": "36 to 40",
        "Days_Policy_Accident": "15 to 30",
        "Days_Policy_Claim": "15 to 30",
        "PastNumberOfClaims": "1",
        "NumberOfSuppliments": "1 to 2",
        "AddressChange_Claim": "1 year",
        "NumberOfCars": "1 vehicle",
        "Make": "BMW",
        "PolicyType": "Sedan - All Perils",
        "MaritalStatus": "Divorced",
        "VehicleCategory": "Sedan",
        "BasePolicy": "All Perils",
        "Month": "Jun",
        "MonthClaimed": "Jul",
        "DayOfWeek": "Wednesday",
        "DayOfWeekClaimed": "Thursday",
        "Deductible": 500,
        "DriverRating": 3,
        "WeekOfMonth": 3,
        "WeekOfMonthClaimed": 1,
    },
    # ---- Claim 4: Elderly driver, rural area, low risk -------------------
    {
        "claim_id": _make_id(),
        "submitted_at": _now_iso(),
        "Age": 68,
        "Sex": "Female",
        "AccidentArea": "Rural",
        "Fault": "Third Party",
        "PoliceReportFiled": "Yes",
        "WitnessPresent": "Yes",
        "AgentType": "Internal",
        "VehiclePrice": "30000 to 39000",
        "AgeOfVehicle": "6 years",
        "AgeOfPolicyHolder": "over 65",
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "PastNumberOfClaims": "none",
        "NumberOfSuppliments": "none",
        "AddressChange_Claim": "4 to 8 years",
        "NumberOfCars": "2 vehicles",
        "Make": "Honda",
        "PolicyType": "Utility - Liability",
        "MaritalStatus": "Widow",
        "VehicleCategory": "Utility",
        "BasePolicy": "Liability",
        "Month": "Sep",
        "MonthClaimed": "Sep",
        "DayOfWeek": "Tuesday",
        "DayOfWeekClaimed": "Thursday",
        "Deductible": 700,
        "DriverRating": 1,
        "WeekOfMonth": 4,
        "WeekOfMonthClaimed": 4,
    },
    # ---- Claim 5: Suspicious — new policy, quick claim, no witnesses -----
    {
        "claim_id": _make_id(),
        "submitted_at": _now_iso(),
        "Age": 29,
        "Sex": "Male",
        "AccidentArea": "Urban",
        "Fault": "Policy Holder",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "AgentType": "External",
        "VehiclePrice": "60000 to 69000",
        "AgeOfVehicle": "new",
        "AgeOfPolicyHolder": "26 to 30",
        "Days_Policy_Accident": "none",
        "Days_Policy_Claim": "none",
        "PastNumberOfClaims": "2 to 4",
        "NumberOfSuppliments": "3 to 5",
        "AddressChange_Claim": "under 6 months",
        "NumberOfCars": "1 vehicle",
        "Make": "Lexus",
        "PolicyType": "Sport - Collision",
        "MaritalStatus": "Single",
        "VehicleCategory": "Sport",
        "BasePolicy": "Collision",
        "Month": "Dec",
        "MonthClaimed": "Dec",
        "DayOfWeek": "Saturday",
        "DayOfWeekClaimed": "Monday",
        "Deductible": 300,
        "DriverRating": 4,
        "WeekOfMonth": 5,
        "WeekOfMonthClaimed": 1,
    },
]
