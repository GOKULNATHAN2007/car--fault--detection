import pandas as pd

# ============================================================
# STEP 1 - LOAD DATASET
# ============================================================

df = pd.read_csv("car_failure_detection_dataset.csv")

print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFault Types:")
print(df["fault"].value_counts())


# ============================================================
# STEP 2 - VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Count each fault type
fault_counts = df["fault"].value_counts()

# Create bar chart
plt.figure(figsize=(10, 5))

sns.barplot(
    x=fault_counts.index,
    y=fault_counts.values
)

plt.title("Car Fault Distribution")
plt.xlabel("Fault Type")
plt.ylabel("Number of Cars")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 3 - INPUT AND TARGET
# ============================================================

X = df.drop("fault", axis=1)

y = df["fault"]


print("\nInput Features (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())


print("\nFeature Names:")
print(X.columns.tolist())


# ============================================================
# STEP 4 - LABEL ENCODING
# ============================================================

from sklearn.preprocessing import LabelEncoder

# Convert fault names into numbers
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)


print("\nEncoded Target:")
print(y_encoded[:10])


print("\nFault Mapping:")

for number, name in enumerate(encoder.classes_):

    print(number, "=", name)


# ============================================================
# STEP 5 - TRAIN TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


print("\nTraining Data:")

print("X_train:", X_train.shape)

print("y_train:", y_train.shape)


print("\nTesting Data:")

print("X_test:", X_test.shape)

print("y_test:", y_test.shape)


# ============================================================
# STEP 6 - TRAIN RANDOM FOREST MODEL
# ============================================================

from sklearn.ensemble import RandomForestClassifier

# Create the model

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

    n_jobs=-1
)


# Train the model

model.fit(

    X_train,

    y_train
)


print(
    "\nModel training completed successfully!"
)


# ============================================================
# STEP 7 - MAKE PREDICTIONS
# ============================================================

# Predict the fault for test data

y_pred = model.predict(

    X_test
)


print("\nActual Values:")

print(y_test)


print("\nPredicted Values:")

print(y_pred)


# ============================================================
# STEP 8 - CALCULATE ACCURACY
# ============================================================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(

    y_test,

    y_pred
)


print("\nModel Accuracy:")

print(

    f"{accuracy * 100:.2f}%"

)


# ============================================================
# STEP 9 - CONFUSION MATRIX
# ============================================================

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


cm = confusion_matrix(

    y_test,

    y_pred
)


print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred,

        target_names=encoder.classes_

    )

)


# ============================================================
# STEP 10 - TEST A NEW CAR
# ============================================================

# IMPORTANT:
# The values MUST follow the same order as X.columns

print("\nCurrent Feature Order:")

print(X.columns.tolist())


# This part is fixed.
# It automatically creates the new car using
# the same column names and order as the training data.

new_car = pd.DataFrame(

    [[
        105,      # engine_temp
        12.3,     # battery_voltage
        2500,     # rpm
        45,       # fuel_pressure
        50,       # oil_pressure
        90,       # coolant_temp
        1,        # vibration
        80000,    # mileage
        8         # age
    ]],

    columns=[
        "engine_temp",
        "battery_voltage",
        "rpm",
        "fuel_pressure",
        "oil_pressure",
        "coolant_temp",
        "vibration",
        "mileage",
        "age"
    ]

)


# Check that training columns and new-car columns match

if list(X.columns) != list(new_car.columns):

    print("\n❌ FEATURE MISMATCH!")

    print("\nTraining Features:")

    print(X.columns.tolist())

    print("\nNew Car Features:")

    print(new_car.columns.tolist())

    print(
        "\nPlease make sure your dataset contains "
        "the same 9 features used by the UI."
    )

else:

    prediction = model.predict(

        new_car
    )


    predicted_fault = encoder.inverse_transform(

        prediction

    )


    print("\nNew Car Prediction:")

    print(
        "Predicted Fault:",
        predicted_fault[0]
    )


# ============================================================
# STEP 11 - SAVE THE MODEL
# ============================================================

import joblib


# Save the NEWLY TRAINED model

joblib.dump(

    model,

    "car_failure_model.pkl"
)


# Save the encoder

joblib.dump(

    encoder,

    "fault_encoder.pkl"
)


print(
    "\nModel saved successfully!"
)


print(
    "Encoder saved successfully!"
)


print(
    "\nModel Feature Names:"
)


print(
    model.feature_names_in_
)


print(
    "\n✅ New model is now ready for Streamlit UI!"
)
