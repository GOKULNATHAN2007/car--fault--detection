"""Train and evaluate the Random Forest model for the classroom demo."""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = "car_failure_detection_dataset.csv"
MODEL_FILE = "car_failure_model.pkl"
ENCODER_FILE = "fault_encoder.pkl"
FEATURES = [
    "engine_temp", "battery_voltage", "rpm", "fuel_pressure", "oil_pressure",
    "coolant_temp", "vibration", "mileage", "age"
]


def main():
    df = pd.read_csv(DATA_FILE)
    required = FEATURES + ["fault"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[required].dropna().copy()
    X = df[FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["fault"].astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, pred) * 100:.2f}%")
    print("Confusion matrix (class order:", list(encoder.classes_), ")")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=encoder.classes_, zero_division=0))
    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)
    print(f"Saved {MODEL_FILE} and {ENCODER_FILE}")
    print("Note: metrics measure this dataset split only; synthetic data performance does not prove real-world accuracy.")

if __name__ == "__main__":
    main()
