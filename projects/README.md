# Data Mining Projects

This directory contains three main data mining projects focused on diabetes management through machine learning and statistical analysis.

## 📋 Project Overview

### 1. Meal Detection Project
**Location**: `meal-detection/`
**Objective**: Predict meal vs. no-meal events from glucose data using machine learning

**Key Features**:
- Feature extraction from glucose time series
- SMOTE oversampling for imbalanced data
- Multiple ML models (Gradient Boosting, Random Forest, SVM, Decision Trees)
- Cross-validation and hyperparameter tuning

### 2. Time Series Analysis Project
**Location**: `time-series-analysis/`
**Objective**: Extract meaningful features from glucose time series data

**Key Features**:
- Statistical metrics computation
- Day/night glucose segmentation
- Glucose variability analysis
- Automated vs. manual mode comparison

### 3. Clustering Validation Project
**Location**: `clustering-validation/`
**Objective**: Validate clustering algorithms for glucose pattern recognition

**Key Features**:
- K-means and DBSCAN clustering
- Purity and entropy metrics
- Ground truth validation
- Performance comparison

## 🚀 Running the Projects

### Prerequisites
Make sure you have the required dependencies installed:
```bash
pip install -r ../../requirements.txt
```

### Project 1: Meal Detection

#### Training the Model
```bash
cd meal-detection
python train-model.py
```

**What it does**:
- Loads CGM and insulin data
- Extracts meal and no-meal segments
- Engineers features from glucose patterns
- Trains multiple ML models
- Saves the best model and preprocessing components

#### Making Predictions
```bash
python predict.py
```

**What it does**:
- Loads the trained model
- Processes new test data
- Makes meal/no-meal predictions
- Saves results to CSV

### Project 2: Time Series Analysis

#### Running the Analysis
```bash
cd time-series-analysis
python time-series-extraction.py
```

**What it does**:
- Loads CGM and insulin data
- Segments data into auto vs. manual modes
- Computes daily glucose metrics
- Analyzes day/night patterns
- Generates statistical summaries

### Project 3: Clustering Validation

#### Running the Validation
```bash
cd clustering-validation
python cluster_validation.py
```

**What it does**:
- Loads glucose and insulin data
- Extracts meal data segments
- Applies K-means and DBSCAN clustering
- Computes validation metrics (SSE, purity, entropy)
- Saves results to CSV

## 📊 Expected Outputs

### Meal Detection Outputs
- **Trained Model**: `model_scaler_sampler.pkl` (contains model, scaler, and sampler)
- **Predictions**: `Result.csv` (meal/no-meal predictions)

### Time Series Analysis Outputs
- **Segmented Data**: Auto vs. manual mode glucose data
- **Statistical Metrics**: Daily glucose statistics
- **Day/Night Analysis**: Temporal pattern analysis

### Clustering Validation Outputs
- **Results**: `Result.csv` (clustering performance metrics)
- **Metrics**: SSE, purity, and entropy for each algorithm

## 🔬 Technical Details

### Data Requirements
All projects require the following datasets in the `../datasets/` directory:
- `CGMData.csv` - Continuous glucose monitoring data
- `InsulinData.csv` - Insulin administration data
- Additional datasets for validation

### Model Performance
- **Meal Detection**: Typically achieves 70-85% accuracy
- **Clustering**: Validated using purity and entropy metrics
- **Time Series**: Provides comprehensive glucose pattern analysis

### Computational Requirements
- **Memory**: 4-8GB RAM recommended for large datasets
- **Processing**: Multi-core processing beneficial for model training
- **Storage**: ~50MB for models and results

## 🎯 Learning Objectives

### Machine Learning Skills
1. **Feature Engineering**: Extract meaningful features from time series
2. **Model Selection**: Compare multiple algorithms
3. **Validation**: Use appropriate metrics for evaluation
4. **Preprocessing**: Handle imbalanced data and missing values

### Domain Knowledge
1. **Diabetes Management**: Understand glucose monitoring
2. **Medical Data**: Handle sensitive healthcare data
3. **Time Series**: Work with temporal data patterns
4. **Statistical Analysis**: Apply statistical methods to real data

## 📝 Project Workflow

### Typical Workflow
1. **Data Loading**: Load CGM and insulin datasets
2. **Preprocessing**: Clean and align temporal data
3. **Feature Extraction**: Engineer relevant features
4. **Model Training**: Train and validate models
5. **Evaluation**: Assess performance using appropriate metrics
6. **Results**: Save and document findings

### Best Practices
- Always validate results with domain experts
- Use appropriate cross-validation strategies
- Document preprocessing steps thoroughly
- Consider ethical implications of medical data analysis

## 🔗 Dependencies

### Python Libraries
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **imbalanced-learn**: Handle imbalanced datasets

### Data Dependencies
- CGM and insulin datasets in `../datasets/`
- Proper data format and structure
- Sufficient data quality for analysis

## 🎯 Expected Outcomes

After completing these projects, you should be able to:
1. Preprocess medical time series data effectively
2. Apply machine learning to classification problems
3. Handle imbalanced datasets using SMOTE
4. Validate clustering algorithms properly
5. Extract meaningful features from time series
6. Interpret results in medical context
7. Document analysis workflows comprehensively

## 📚 Additional Resources

### Documentation
- Check `../documentation/` for detailed guides
- Review PDF files for theoretical background
- Consult project-specific README files

### Data Sources
- Datasets are in `../datasets/` directory
- Results are saved in project directories
- Models are saved as pickle files

---

**Note**: These projects involve medical data analysis. Results should be interpreted carefully and not used for clinical decisions without proper validation.
