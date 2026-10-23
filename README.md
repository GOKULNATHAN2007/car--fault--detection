# car--fault--detection

A machine learning project that predicts car component faults from sensor readings using a Random Forest classifier. It includes an interactive Streamlit web app with a virtual vehicle diagnostic display.

FEATURES

- Random Forest classification of car faults: No Fault, Warning, Fault
- About 95 percent model accuracy on the test split
- Interactive web UI built with Streamlit
- Virtual car diagram that colours each component by its status
- Single-vehicle analysis and batch prediction from CSV
- Model information and About pages

TECH STACK

- Python 3.8 or higher
- Pandas and NumPy
- Scikit-learn
- Matplotlib, Seaborn and Plotly
- Joblib
- Streamlit

PROJECT STRUCTURE

setup.py                            Creates dataset and trains the model
car_fault_ui.py                     Streamlit web application
requirements.txt                    Python dependencies
RUN_APP.py                          One-command setup and launch
car_failure_detection_dataset.csv   Generated dataset
car_failure_model.pkl               Trained model (generated)
fault_encoder.pkl                   Label encoder (generated)
README.txt                          This file

INSTALLATION

1. Clone the repository:
   git clone https://github.com/your-username/your-repo.git
2. Go into the folder:
   cd your-repo
3. Install the dependencies:
   pip install -r requirements.txt

USAGE

Option 1: One command (recommended)
   python RUN_APP.py
This installs dependencies, generates the dataset, trains the model and launches the app.

Option 2: Step by step
   python setup.py
   streamlit run car_fault_ui.py


ML PIPELINE

1. Import the CSV dataset and check its structure and missing values
2. Explore fault distribution with a bar chart
3. Separate features (X) and target (y)
4. Encode the fault labels with LabelEncoder
5. Split into 80 percent training and 20 percent test data (stratified)
6. Train a RandomForestClassifier with 100 trees
7. Predict on the test set
8. Evaluate accuracy, confusion matrix and classification report
9. Test a new vehicle's sensor readings
10. Save the model and encoder with joblib

INPUT FEATURES

engine_temp       (°C)    Engine temperature
battery_voltage   (V)     Battery voltage
rpm               (RPM)   Engine speed
fuel_pressure     (psi)   Fuel pressure
oil_pressure      (psi)   Oil pressure
coolant_temp      (°C)    Coolant temperature
vibration         (0-2)   0 = Normal, 1 = High, 2 = Very High
mileage           (km)    Distance travelled
age               (years) Vehicle age

OUTPUT CATEGORIES

No Fault    Vehicle operating normally
Warning     Minor issue, servicing recommended
Fault       Critical issue, immediate attention required

SAMPLE INPUT

engine_temp=95, battery_voltage=12.6, rpm=2500, fuel_pressure=45,
oil_pressure=50, coolant_temp=90, vibration=0, mileage=50000, age=5

MODEL PERFORMANCE

- Accuracy: about 95 percent (verify the exact figure by running setup.py)
- Algorithm: Random Forest, 100 estimators, random_state=42
- Split: 80 percent train, 20 percent test, stratified

FUTURE IMPROVEMENTS

- Use a real vehicle sensor dataset instead of synthetic data
- Add a feature importance chart to the UI
- Add live sensor input via OBD-II
- Deploy the app on Streamlit Community Cloud

ACADEMIC NOTE

This is a college project. The dataset is generated synthetically for demonstration purposes.

LICENSE

For educational use.
