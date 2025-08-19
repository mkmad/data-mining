# Data Mining Datasets

This directory contains the datasets used for diabetes management analysis through Continuous Glucose Monitoring (CGM) and insulin data.

## 📊 Dataset Overview

### Primary Datasets

#### CGM Data (Continuous Glucose Monitoring)
- **`CGMData.csv`** - Main CGM dataset with glucose readings
- **`CGMData_1.csv`** - Additional CGM dataset for validation
- **`CGM_patient2.csv`** - CGM data for a second patient

**Purpose**: Real-time glucose level monitoring for diabetes management

**Features**:
- **Date**: Date of measurement
- **Time**: Time of measurement
- **Sensor Glucose (mg/dL)**: Glucose level in milligrams per deciliter
- **Index**: Sequential measurement index

**Frequency**: Measurements every 5 minutes
**Applications**: Diabetes management, treatment optimization, meal detection

#### Insulin Data
- **`InsulinData.csv`** - Main insulin administration dataset
- **`InsulinData_2.csv`** - Additional insulin dataset for validation
- **`Insulin_patient2.csv`** - Insulin data for a second patient

**Purpose**: Records of insulin administration and carbohydrate intake

**Features**:
- **Date**: Date of insulin administration
- **Time**: Time of insulin administration
- **BWZ Carb Input (grams)**: Carbohydrate intake in grams
- **Alarm**: System alarms and mode indicators
- **Index**: Sequential administration index

**Applications**: Meal detection, insulin therapy optimization, treatment analysis

### Results Datasets

#### Analysis Results
- **`Results.csv`** - Clustering validation results
- **`Final_Results.csv`** - Final project results and predictions

**Purpose**: Store outputs from various analysis projects

## 🔬 Data Characteristics

### CGM Data Characteristics
- **Temporal Resolution**: 5-minute intervals
- **Glucose Range**: Typically 40-400 mg/dL
- **Missing Data**: Some gaps due to sensor issues
- **Data Quality**: High accuracy for diabetes management

### Insulin Data Characteristics
- **Event-Based**: Records when insulin is administered
- **Carbohydrate Tracking**: Links meals to insulin doses
- **Mode Tracking**: Monitors automated vs. manual insulin delivery
- **Temporal Alignment**: Synchronized with CGM data

## 📋 Data Preprocessing Notes

### Common Preprocessing Steps
1. **Time Alignment**: Synchronize CGM and insulin timestamps
2. **Missing Data Handling**: Interpolate or remove missing glucose readings
3. **Outlier Detection**: Remove physiologically impossible glucose values
4. **Feature Engineering**: Extract meal windows and glucose patterns

### Data Quality Considerations
- **Sensor Accuracy**: CGM sensors have ±15% accuracy
- **Calibration**: Regular calibration required for accuracy
- **Environmental Factors**: Temperature and humidity affect readings
- **Patient Factors**: Individual variations in glucose metabolism

## 🎯 Usage Guidelines

### For Meal Detection Projects
- Use `CGMData.csv` and `InsulinData.csv` as primary datasets
- Extract meal windows based on carbohydrate input
- Align glucose data with meal events

### For Time Series Analysis
- Focus on `CGMData.csv` for continuous glucose patterns
- Consider day/night segmentation
- Extract statistical features from glucose time series

### For Clustering Validation
- Use both CGM and insulin data for comprehensive analysis
- Consider patient-specific patterns
- Validate clustering results against known meal events

## 🔒 Privacy and Ethics

### Data Privacy
- All data is anonymized for research purposes
- No personally identifiable information included
- Data used only for educational and research purposes

### Ethical Considerations
- Medical data requires careful handling
- Results should not be used for clinical decisions without validation
- Respect patient privacy and data confidentiality

## 📝 Data Format

### CSV Format
All datasets are in CSV format with the following structure:
```csv
Index,Date,Time,Feature1,Feature2,...
1,2023-01-01,00:00:00,value1,value2,...
2,2023-01-01,00:05:00,value1,value2,...
```

### File Sizes
- **CGM Data**: ~4MB each (large due to high temporal resolution)
- **Insulin Data**: ~4MB each (comprehensive administration records)
- **Results**: Small files (<2KB) containing analysis outputs

## 🔗 Related Projects

These datasets are used in the following projects:
- **Meal Detection**: `../projects/meal-detection/`
- **Time Series Analysis**: `../projects/time-series-analysis/`
- **Clustering Validation**: `../projects/clustering-validation/`

## 📚 References

For more information about diabetes data analysis:
- [Continuous Glucose Monitoring](https://www.diabetes.org/diabetes/continuous-glucose-monitoring)
- [Insulin Therapy Guidelines](https://www.diabetes.org/diabetes/medication-management/insulin-other-injectables)
- [Data Mining in Healthcare](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4287079/)

---

**Note**: These datasets are for educational and research purposes only. Always consult healthcare professionals for medical advice and treatment decisions.
