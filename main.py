# -*- coding: utf-8 -*-

import pandas as pd

def load_data(cgm_filepath, insulin_filepath):
    cgm_data = pd.read_csv(cgm_filepath, low_memory=False).filter(['Index','Date','Time','Sensor Glucose (mg/dL)'])
    insulin_data = pd.read_csv(insulin_filepath, low_memory=False).filter(['Index','Date','Time','Alarm'])
    return cgm_data, insulin_data

def find_segment_data_date(insulin_data):
    # Extract Manual and Auto Mode time filter
    seg_filter = (insulin_data['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF')
    seg_date = insulin_data[seg_filter].iloc[-1,-1]    
    return seg_date

def filter_glucose_data(glucose_data):
    # Set 'DateTime' as the index if it isn't already
    if not glucose_data.index.name == 'Datetime':
        glucose_data.set_index('Datetime', inplace=True)
    
    sufficient_data_threshold = 231  
    daily_data_count = glucose_data['Sensor Glucose (mg/dL)'].notnull().resample('D').sum()
    insufficient_data_dates = daily_data_count[daily_data_count < sufficient_data_threshold].index.tolist()
    drop_list = []
    # Drop days with insufficient data
    for date in insufficient_data_dates:
        drop_list.append(str(date)[:10])    
    for dl in drop_list:
        glucose_data.drop(glucose_data[dl].index.tolist(), inplace=True)
    return glucose_data

def segment_data(cgm_data, insulin_data):
    # Segment Auto vs Manual Data
    # Ensure DateTime conversion is done correctly
    cgm_data['Datetime'] = cgm_data['Date'] + ' ' + cgm_data['Time']
    cgm_data['Time'] = cgm_data['Date'] + ' ' + cgm_data['Time']
    cgm_data['Datetime'] = pd.to_datetime(cgm_data['Datetime'])
    cgm_data['Date'] = pd.to_datetime(cgm_data['Date'])
    cgm_data['Time'] = pd.to_datetime(cgm_data['Time'])
    insulin_data['Datetime'] = insulin_data['Date'] + ' ' + insulin_data['Time']
    insulin_data['Time'] = insulin_data['Date'] + ' ' + insulin_data['Time']
    insulin_data['Datetime'] = pd.to_datetime(insulin_data['Datetime'])
    insulin_data['Date'] = pd.to_datetime(insulin_data['Date'])
    insulin_data['Time'] = pd.to_datetime(insulin_data['Time'])
    
    # Find the Segment date filter
    seg_date = find_segment_data_date(insulin_data=insulin_data)

    cgm_manual_mode_data = cgm_data[cgm_data['Datetime'] <= seg_date].copy()
    cgm_auto_mode_data = cgm_data[cgm_data['Datetime'] > seg_date].copy()
    cgm_manual_mode_data.set_index('Datetime', inplace=True)
    cgm_auto_mode_data.set_index('Datetime', inplace=True)

    # Apply filters
    cgm_manual_mode_data = filter_glucose_data(cgm_manual_mode_data)
    cgm_auto_mode_data = filter_glucose_data(cgm_auto_mode_data)

    return cgm_manual_mode_data, cgm_auto_mode_data


def compute_glucose_metrics(cgm_data):
    """
    Compute metrics for AUTO Mode based on provided DataFrame.

    Args:
    - cgm_data: DataFrame containing glucose data.

    Returns:
    - DataFrame with computed metrics.
    """
    # Initialize an empty DataFrame to store metrics

    mean_glucose = cgm_data['Sensor Glucose (mg/dL)'].resample('D').mean()
    date_index = mean_glucose.index.tolist()
    metrics = pd.DataFrame()

    for i in range(len(date_index)):
        # Filter data for the current day
        day_data = cgm_data[cgm_data['Date'] == date_index[i]].copy()
        day_data['Time'] = pd.to_datetime(day_data['Time'])  # Convert 'Time' column to datetime
        
        day_data.fillna(mean_glucose[date_index[i]], inplace=True)
        
        date = str(date_index[i])[:10]
        dateDay6am = pd.Timestamp(date + ' 06:00:00')
        dateDay12am = pd.Timestamp(date + ' 23:59:59')
        dateNight12am = pd.Timestamp(date + ' 00:00:00')
        
        filt = (day_data['Time'] >= dateDay6am) & (day_data['Time'] <= dateDay12am)
        day = day_data[filt].copy()
        
        filt = (day_data['Time'] >= dateNight12am) & (day_data['Time'] < dateDay6am)
        night = day_data[filt].copy()

        # Compute metrics for night
        night_m1 = len(night[night['Sensor Glucose (mg/dL)'] > 180]) / 2.88
        night_m2 = len(night[night['Sensor Glucose (mg/dL)'] > 250]) / 2.88
        night_m3 = len(night[(night['Sensor Glucose (mg/dL)'] >= 70) & (night['Sensor Glucose (mg/dL)'] <= 180)]) / 2.88
        night_m4 = len(night[(night['Sensor Glucose (mg/dL)'] >= 70) & (night['Sensor Glucose (mg/dL)'] <= 150)]) / 2.88
        night_m5 = len(night[night['Sensor Glucose (mg/dL)'] < 70]) / 2.88
        night_m6 = len(night[night['Sensor Glucose (mg/dL)'] < 54]) / 2.88
        
        # Compute metrics for day
        day_m1 = len(day[day['Sensor Glucose (mg/dL)'] > 180]) / 2.88
        day_m2 = len(day[day['Sensor Glucose (mg/dL)'] > 250]) / 2.88
        day_m3 = len(day[(day['Sensor Glucose (mg/dL)'] >= 70) & (day['Sensor Glucose (mg/dL)'] <= 180)]) / 2.88
        day_m4 = len(day[(day['Sensor Glucose (mg/dL)'] >= 70) & (day['Sensor Glucose (mg/dL)'] <= 150)]) / 2.88
        day_m5 = len(day[day['Sensor Glucose (mg/dL)'] < 70]) / 2.88
        day_m6 = len(day[day['Sensor Glucose (mg/dL)'] < 54]) / 2.88
        
        # Compute metrics for whole day
        whole_day_m1 = len(day_data[day_data['Sensor Glucose (mg/dL)'] > 180]) / 2.88
        whole_day_m2 = len(day_data[day_data['Sensor Glucose (mg/dL)'] > 250]) / 2.88
        whole_day_m3 = len(day_data[(day_data['Sensor Glucose (mg/dL)'] >= 70) & (day_data['Sensor Glucose (mg/dL)'] <= 180)]) / 2.88
        whole_day_m4 = len(day_data[(day_data['Sensor Glucose (mg/dL)'] >= 70) & (day_data['Sensor Glucose (mg/dL)'] <= 150)]) / 2.88
        whole_day_m5 = len(day_data[day_data['Sensor Glucose (mg/dL)'] < 70]) / 2.88
        whole_day_m6 = len(day_data[day_data['Sensor Glucose (mg/dL)'] < 54]) / 2.88
        
        # Store metrics in DataFrame
        metrics_list = [night_m1, night_m2, night_m3, night_m4, night_m5, night_m6, 
                           day_m1, day_m2, day_m3, day_m4, day_m5, day_m6, 
                           whole_day_m1, whole_day_m2, whole_day_m3, whole_day_m4, whole_day_m5, whole_day_m6]
        metrics[i] = metrics_list

    return metrics


def save_combined_metrics_to_csv(manual_data, auto_data, filename='Result.csv'):
    """
    Aggregates metrics for both manual and auto mode data into mean values,
    then combines them into a single DataFrame to save to a CSV file.
    This version ensures only two rows are output, corresponding to manual and auto modes.
    """
    # Calculate means for manual and auto data
    data_result = [list(manual_data.mean(axis=1))]
    data_result.append(list(auto_data.mean(axis=1)))

    # Create DataFrame with means
    output = pd.DataFrame(data_result)

    # Save DataFrame to CSV without index and header
    output.to_csv(filename, index=False, header=False)


if __name__ == "__main__":
    cgm_filepath = 'CGMData.csv'
    insulin_filepath = 'InsulinData.csv'
    output_filepath = 'Result.csv'
    
    # Load and process the data
    cgm_data, insulin_data = load_data(cgm_filepath, insulin_filepath)
    cgm_manual_mode_data, cgm_auto_mode_data = segment_data(cgm_data, insulin_data)
    
    auto_cgm_metrics = compute_glucose_metrics(cgm_data=cgm_auto_mode_data)
    manual_cgm_metrics = compute_glucose_metrics(cgm_data=cgm_manual_mode_data)

    save_combined_metrics_to_csv(
        manual_data=manual_cgm_metrics, 
        auto_data=auto_cgm_metrics,
        filename=output_filepath
    )
