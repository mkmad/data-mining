import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fft import fft
import scipy.stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle

# Function to read the CSV files
def read_csv_files(insulin_file, cgm_file):
    insulin_data = pd.read_csv(insulin_file, parse_dates=['Date'], usecols=['Date', 'BWZ Carb Input (grams)'])
    cgm_data = pd.read_csv(cgm_file, parse_dates=['Date'], usecols=['Date', 'Sensor Glucose (mg/dL)'])
    return insulin_data, cgm_data

# Function to extract meal data
def extract_meal_data(insulin_data, cgm_data):
    meal_data_matrix = []
    insulin_sorted = insulin_data.notnull()
    insulin_sorted = insulin_data.sort_values('Date')
    for index, row in insulin_sorted.iterrows():
        if row['BWZ Carb Input (grams)'] is not pd.NaT:
            if row['BWZ Carb Input (grams)'] > 0:
                tm = row['Date']
                start_time = tm - timedelta(minutes=30)
                end_time = tm + timedelta(hours=2)
                mask = (cgm_data['Date'] >= start_time) & (cgm_data['Date'] <= end_time)
                data_slice = cgm_data.loc[mask]['Sensor Glucose (mg/dL)'].tolist()
                if len(data_slice) >= 24:
                    meal_data_matrix.append(data_slice[:24])
    return pd.DataFrame(meal_data_matrix)

# Function to extract no meal data
def extract_no_meal_data(insulin_data, cgm_data):
    no_meal_data_matrix = []
    insulin_sorted = insulin_data.sort_values('Date')
    for index in range(len(insulin_sorted) - 1):
        current_time = insulin_sorted.iloc[index]['Date']
        next_time = insulin_sorted.iloc[index + 1]['Date']
        if next_time - current_time > timedelta(hours=4):
            start_time = current_time + timedelta(hours=2)
            end_time = start_time + timedelta(hours=2)
            mask = (cgm_data['Date'] >= start_time) & (cgm_data['Date'] <= end_time)
            data_slice = cgm_data.loc[mask]['Sensor Glucose (mg/dL)'].tolist()
            if len(data_slice) >= 24:
                no_meal_data_matrix.append(data_slice[:24])
    return pd.DataFrame(no_meal_data_matrix)

# Function to extract features
def extract_features(data_matrix):
    features_list = []

    for index, row in data_matrix.iterrows():
        row_array = row.to_numpy()

        # Statistical features
        mean_val = np.mean(row_array)
        std_val = np.std(row_array)
        max_val = np.max(row_array)
        min_val = np.min(row_array)
        median_val = np.median(row_array)
        var_val = np.var(row_array)
        range_val = max_val - min_val
        skew_val = scipy.stats.skew(row_array)
        kurtosis_val = scipy.stats.kurtosis(row_array)

        # Frequency-domain features: Calculate FFT and select the first few components as features
        fft_values = fft(row_array)
        fft_magnitude = np.abs(fft_values)
        fft_features = fft_magnitude[:15]  # Make sure this count contributes correctly to the total feature count

        # Concatenate all features
        features = [
            mean_val, std_val, max_val, min_val, median_val, var_val, range_val,
            skew_val, kurtosis_val
        ] + fft_features.tolist()

        # Ensure we have exactly 24 features
        assert len(features) == 24, f"Expected 24 features, but got {len(features)}"
        features_list.append(features)

    return pd.DataFrame(features_list, columns=[f'feature_{i+1}' for i in range(24)])


# Function to train the RandomForest model
def train_model(features, meal_features_count, no_meal_features_count):

    X = features
    y = [1] * meal_features_count + [0] * no_meal_features_count
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123)

    model = RandomForestClassifier(n_estimators=20, min_samples_split=15, min_samples_leaf=5,
                                   max_depth=10, random_state=123, class_weight='balanced')
    model.fit(X_train, y_train)

    return model

# Function to retrieve the trained model
def consolidate_features(insulin_file='InsulinData.csv', cgm_file='CGMData.csv'):
    insulin_data, cgm_data = read_csv_files(insulin_file, cgm_file)
    meal_data_matrix = extract_meal_data(insulin_data, cgm_data)
    no_meal_data_matrix = extract_no_meal_data(insulin_data, cgm_data)

    meal_features = extract_features(meal_data_matrix)
    no_meal_features = extract_features(no_meal_data_matrix)
    consolidate_features = meal_features.append(no_meal_features)

    return consolidate_features, len(meal_features), len(no_meal_features)

# Fill NaN values with the column mean or median before training
def fill_na(data):
    return data.fillna(data.mean())

# Main function orchestrating the process
def main():
    patient1_insulin_file = 'InsulinData.csv'
    patient1_cgm_file = 'CGMData.csv'

    patient2_cgm_file = 'CGM_patient2.csv'
    patient2_insulin_file = 'Insulin_patient2.csv'

    # Train the model and save the results
    patient_1_features, patient_1_num_meal_data, patient_1_num_no_meal_data = consolidate_features(patient1_insulin_file, patient1_cgm_file)
    patient_2_features, patient_2_num_meal_data, patient_2_num_no_meal_data = consolidate_features(patient2_insulin_file, patient2_cgm_file)
    patient_1_features = fill_na(patient_1_features)
    patient_2_features = fill_na(patient_2_features)
    features_to_train = pd.concat([patient_1_features, patient_2_features], ignore_index=True)
    # Train the model and save the results
    model = train_model(
        features_to_train, 
        patient_1_num_meal_data + patient_2_num_meal_data, 
        patient_1_num_no_meal_data + patient_2_num_no_meal_data
    )


    # Save the trained model and scaler to a pickle file
    with open('model_scaler.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
