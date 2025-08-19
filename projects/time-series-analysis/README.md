# Time Series Analysis Project

This project focuses on extracting meaningful features from Continuous Glucose Monitoring (CGM) time series data, with particular emphasis on automated vs. manual insulin delivery modes and day/night glucose patterns.

## 🎯 Project Objective

Analyze glucose time series data to extract statistical features, compare automated vs. manual insulin delivery modes, and identify temporal patterns in glucose variability.

## 📊 Methodology

### Data Processing Pipeline

#### 1. Data Loading and Preprocessing
```python
def load_data(cgm_filepath, insulin_filepath):
    # Load CGM and insulin data
    # Filter relevant columns
    # Return synchronized datasets
```

**Data Sources**:
- **CGM Data**: Continuous glucose monitoring readings
- **Insulin Data**: Insulin pump mode information and alarms

#### 2. Mode Segmentation
```python
def segment_data(cgm_data, insulin_data):
    # Identify automated vs. manual mode transition
    # Segment data into two periods
    # Apply data quality filters
```

**Segmentation Criteria**:
- **Manual Mode**: Before automated insulin delivery activation
- **Automated Mode**: After automated insulin delivery activation
- **Transition Point**: Identified by 'AUTO MODE ACTIVE PLGM OFF' alarm

#### 3. Data Quality Filtering
```python
def filter_glucose_data(glucose_data):
    # Remove days with insufficient data
    # Ensure minimum 231 readings per day
    # Maintain data continuity
```

**Quality Thresholds**:
- **Minimum Daily Readings**: 231 (95% of expected 5-minute intervals)
- **Data Continuity**: Remove incomplete days
- **Temporal Alignment**: Ensure proper timestamp synchronization

#### 4. Glucose Metrics Computation
```python
def compute_glucose_metrics(cgm_data):
    # Calculate daily statistical metrics
    # Separate day/night analysis
    # Compute glucose variability measures
```

**Metrics Calculated**:

**Day Metrics (6 AM - 12 AM)**:
- **M1**: Percentage of time > 180 mg/dL
- **M2**: Percentage of time > 250 mg/dL
- **M3**: Percentage of time 70-180 mg/dL
- **M4**: Percentage of time 70-150 mg/dL
- **M5**: Percentage of time < 70 mg/dL

**Night Metrics (12 AM - 6 AM)**:
- **M6**: Percentage of time > 180 mg/dL
- **M7**: Percentage of time > 250 mg/dL
- **M8**: Percentage of time 70-180 mg/dL
- **M9**: Percentage of time 70-150 mg/dL
- **M10**: Percentage of time < 70 mg/dL

**Overall Metrics**:
- **M11**: Mean glucose level
- **M12**: Standard deviation of glucose
- **M13**: Coefficient of variation
- **M14**: Glucose variability index

## 🚀 Usage

### Prerequisites
```bash
pip install pandas numpy scipy
```

### Running the Analysis
```bash
python time-series-extraction.py
```

**Input Requirements**:
- `../datasets/CGMData.csv` - Glucose monitoring data
- `../datasets/InsulinData.csv` - Insulin pump data

**Output**:
- Segmented glucose data (manual vs. automated mode)
- Daily glucose metrics
- Statistical summaries and comparisons

## 📈 Analysis Components

### 1. Mode Comparison Analysis
**Objective**: Compare glucose control between automated and manual insulin delivery

**Key Metrics**:
- **Time in Range**: Percentage of readings in target range (70-180 mg/dL)
- **Hyperglycemia**: Time above 180 mg/dL and 250 mg/dL
- **Hypoglycemia**: Time below 70 mg/dL
- **Glucose Variability**: Standard deviation and coefficient of variation

### 2. Day/Night Pattern Analysis
**Objective**: Identify diurnal variations in glucose control

**Segmentation**:
- **Day Period**: 6:00 AM - 11:59 PM
- **Night Period**: 12:00 AM - 5:59 AM

**Analysis Focus**:
- **Circadian Patterns**: Natural glucose variations
- **Sleep Effects**: Impact of sleep on glucose control
- **Meal Timing**: Day vs. night meal patterns

### 3. Statistical Feature Extraction
**Objective**: Extract comprehensive statistical features for further analysis

**Feature Categories**:
- **Central Tendency**: Mean, median glucose levels
- **Variability**: Standard deviation, coefficient of variation
- **Distribution**: Percentiles, range, skewness
- **Temporal**: Day/night differences, trend analysis

## 🔬 Technical Details

### Data Requirements
- **Temporal Resolution**: 5-minute intervals
- **Duration**: Minimum 7 days of continuous data
- **Quality**: <5% missing values per day
- **Synchronization**: Aligned CGM and insulin timestamps

### Processing Steps
1. **Data Loading**: Read CSV files and parse timestamps
2. **Mode Detection**: Identify automated/manual mode transition
3. **Segmentation**: Split data into mode-specific periods
4. **Quality Control**: Remove insufficient data days
5. **Metric Calculation**: Compute daily glucose statistics
6. **Analysis**: Compare modes and temporal patterns

### Statistical Methods
- **Descriptive Statistics**: Mean, standard deviation, percentiles
- **Time Series Analysis**: Trend detection, seasonality
- **Comparative Analysis**: Paired t-tests, effect sizes
- **Visualization**: Time series plots, box plots, heatmaps

## 📊 Expected Results

### Mode Comparison Results
- **Automated Mode**: Typically shows improved glucose control
- **Manual Mode**: Baseline glucose control patterns
- **Improvement Metrics**: Quantified benefits of automation

### Temporal Pattern Results
- **Day Patterns**: Higher variability, meal-related peaks
- **Night Patterns**: Lower variability, fasting levels
- **Circadian Effects**: Natural glucose rhythm identification

### Statistical Summary
- **Daily Metrics**: 14 comprehensive glucose measures
- **Mode Differences**: Statistical significance testing
- **Variability Analysis**: Glucose stability assessment

## 🔧 Customization

### Metric Customization
Modify `compute_glucose_metrics()` to:
- **Adjust Thresholds**: Change glucose target ranges
- **Add Metrics**: Include additional statistical measures
- **Modify Time Windows**: Change day/night definitions
- **Include External Factors**: Activity, meals, medication

### Analysis Parameters
Customize analysis settings:
- **Quality Thresholds**: Minimum data requirements
- **Time Segmentation**: Day/night period definitions
- **Statistical Methods**: Choice of statistical tests
- **Output Format**: Results presentation style

### Data Preprocessing
Adjust preprocessing steps:
- **Missing Data**: Interpolation vs. removal
- **Outlier Detection**: Statistical vs. clinical thresholds
- **Smoothing**: Moving averages, filtering
- **Normalization**: Standardization methods

## 📝 Best Practices

### Data Quality Assurance
- **Sensor Calibration**: Ensure accurate glucose readings
- **Temporal Alignment**: Synchronize all data sources
- **Completeness**: Verify sufficient data coverage
- **Consistency**: Check for systematic errors

### Analysis Methodology
- **Statistical Rigor**: Use appropriate statistical tests
- **Clinical Relevance**: Focus on clinically meaningful metrics
- **Reproducibility**: Document all analysis steps
- **Validation**: Cross-validate findings

### Interpretation Guidelines
- **Clinical Context**: Consider medical implications
- **Individual Variation**: Account for patient differences
- **External Factors**: Consider lifestyle and medication effects
- **Limitations**: Acknowledge analysis constraints

## 🔗 Dependencies

### Required Libraries
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
```

### Data Dependencies
- CGM and insulin datasets in `../datasets/`
- Proper timestamp formatting
- Sufficient data quality and duration

## 🎯 Clinical Applications

### Diabetes Management
- **Treatment Optimization**: Adjust insulin therapy based on patterns
- **Technology Assessment**: Evaluate automated insulin delivery systems
- **Patient Education**: Identify individual glucose patterns
- **Clinical Decision Support**: Guide treatment decisions

### Research Applications
- **Clinical Trials**: Evaluate new diabetes technologies
- **Population Studies**: Analyze glucose patterns across groups
- **Predictive Modeling**: Develop glucose prediction algorithms
- **Quality Improvement**: Monitor diabetes care quality

## 📚 Additional Resources

### Documentation
- Check `../documentation/time-series-extraction.pdf` for detailed methodology
- Review clinical guidelines for glucose target ranges
- Consult diabetes management best practices

### Related Projects
- **Meal Detection**: `../meal-detection/` for meal pattern analysis
- **Clustering Validation**: `../clustering-validation/` for pattern recognition

---

**Note**: This analysis is for educational and research purposes. Clinical applications require validation and should be interpreted by healthcare professionals.
