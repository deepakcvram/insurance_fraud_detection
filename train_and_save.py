import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

def train_and_export():
    print("Loading dataset...")
    df = pd.read_csv("dataset.csv")

    # Drop non-predictive columns
    df.drop(columns=['PolicyNumber', 'RepNumber', 'Year'], inplace=True, errors='ignore')

    # Data Cleaning: Drop corrupted rows
    df = df[df['DayOfWeekClaimed'] != '0']

    # Impute missing Age (0 values) to 16
    df.loc[df['Age'] == 0, 'Age'] = 16

    # Define binary columns and mappings
    binary_cols = {
        'Sex': {'Male': 1, 'Female': 0},
        'AccidentArea': {'Urban': 1, 'Rural': 0},
        'Fault': {'Policy Holder': 1, 'Third Party': 0},
        'PoliceReportFiled': {'Yes': 1, 'No': 0},
        'WitnessPresent': {'Yes': 1, 'No': 0},
        'AgentType': {'External': 1, 'Internal': 0}
    }

    # Map binary columns
    for col, mapping in binary_cols.items():
        df[col] = df[col].map(mapping)

    # Define ordinal columns and mappings
    ordinal_mappings = {
        'VehiclePrice': {
            'less than 20000': 0, '20000 to 29000': 1, '30000 to 39000': 2,
            '40000 to 59000': 3, '60000 to 69000': 4, 'more than 69000': 5
        },
        'AgeOfVehicle': {
            'new': 0, '2 years': 1, '3 years': 2, '4 years': 3,
            '5 years': 4, '6 years': 5, '7 years': 6, 'more than 7': 7
        },
        'AgeOfPolicyHolder': {
            '16 to 17': 0, '18 to 20': 1, '21 to 25': 2, '26 to 30': 3,
            '31 to 35': 4, '36 to 40': 5, '41 to 50': 6, '51 to 65': 7, 'over 65': 8
        },
        'Days_Policy_Accident': {
            'none': 0, '1 to 7': 1, '8 to 15': 2, '15 to 30': 3, 'more than 30': 4
        },
        'Days_Policy_Claim': {
            'none': 0, '8 to 15': 1, '15 to 30': 2, 'more than 30': 3
        },
        'PastNumberOfClaims': {
            'none': 0, '1': 1, '2 to 4': 2, 'more than 4': 3
        },
        'NumberOfSuppliments': {
            'none': 0, '1 to 2': 1, '3 to 5': 2, 'more than 5': 3
        },
        'AddressChange_Claim': {
            'no change': 0, 'under 6 months': 1, '1 year': 2,
            '2 to 3 years': 3, '4 to 8 years': 4
        },
        'NumberOfCars': {
            '1 vehicle': 0, '2 vehicles': 1, '3 to 4': 2,
            '5 to 8': 3, 'more than 8': 4
        }
    }

    # Map ordinal columns
    for col, mapping in ordinal_mappings.items():
        df[col] = df[col].map(mapping)

    # Separate features and target
    x = df.drop(columns=['FraudFound_P'])
    y = df['FraudFound_P']

    # Train-test split (80/20, stratified)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # One-Hot Encode nominal columns
    nominal_cols = ['Make', 'PolicyType', 'MaritalStatus', 'VehicleCategory', 'BasePolicy', 
                    'Month', 'MonthClaimed', 'DayOfWeek', 'DayOfWeekClaimed']

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoder.fit(x_train[nominal_cols])

    # Transform nominal columns for train set
    encoded_cats_train = encoder.transform(x_train[nominal_cols])
    encoded_cols = encoder.get_feature_names_out(nominal_cols)
    encoded_df_train = pd.DataFrame(encoded_cats_train, columns=encoded_cols, index=x_train.index)
    x_train_preprocessed = x_train.drop(columns=nominal_cols).join(encoded_df_train)

    # Transform nominal columns for test set
    encoded_cats_test = encoder.transform(x_test[nominal_cols])
    encoded_df_test = pd.DataFrame(encoded_cats_test, columns=encoded_cols, index=x_test.index)
    x_test_preprocessed = x_test.drop(columns=nominal_cols).join(encoded_df_test)

    print(f"Train dimensions: {x_train_preprocessed.shape}, Test dimensions: {x_test_preprocessed.shape}")

    # Calculate scale_pos_weight for handling imbalance
    num_neg = len(y_train[y_train == 0])
    num_pos = len(y_train[y_train == 1])
    scale_pos_weight = num_neg / num_pos
    print(f"Negative samples: {num_neg}, Positive samples: {num_pos}, Scale Pos Weight: {scale_pos_weight:.2f}")

    # Train final classifier using XGBoost
    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(x_train_preprocessed, y_train)

    # Get predictions/probabilities on test set for threshold tuning in app
    y_test_probs = model.predict_proba(x_test_preprocessed)[:, 1]

    # Save outputs
    print("Exporting model, encoder, and metadata...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    metadata = {
        'binary_cols': binary_cols,
        'ordinal_mappings': ordinal_mappings,
        'nominal_cols': nominal_cols,
        'feature_columns': list(x_train_preprocessed.columns),
        'y_test': y_test.to_numpy(),
        'y_test_probs': y_test_probs
    }
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("Assets successfully saved to workspace!")

if __name__ == "__main__":
    train_and_export()
