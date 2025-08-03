# Predictive Maintenance Project

## 🌐 Try the App

Experience the predictive maintenance dashboard live: [https://mob-369-predictive-maintenance-app-nof2kx.streamlit.app/](https://mob-369-predictive-maintenance-app-nof2kx.streamlit.app/)

## 🎓 Project Background

This project is part of an IBM internship program, showcasing the practical application of IBM Watson Studio's AutoAI capabilities in industrial predictive maintenance scenarios.


## 📋 Project Overview

A machine learning project for predictive maintenance using IBM Watson Studio's AutoAI capabilities. This project analyzes equipment sensor data to predict potential failures and maintenance needs, helping organizations transition from reactive to proactive maintenance strategies.

This project leverages machine learning to predict equipment failures based on various operational parameters including temperature, rotational speed, torque, and tool wear. The predictive model helps maintenance teams identify potential issues before they lead to costly equipment downtime.

### Key Features

- **Predictive Analytics**: Machine learning models to forecast equipment failures
- **AutoAI Integration**: Automated machine learning pipeline using IBM Watson Studio
- **Incremental Learning**: Support for continuous model improvement with new data
- **Real-time Monitoring**: Capabilities for ongoing equipment health assessment
- **Multiple Failure Types**: Detection of various failure modes (Heat Dissipation, Power, Overstrain, Tool Wear, Random failures)

## 🏗️ Project Structure

```
Predictive-Maintenance/
├── README.md
├── project.json                                    # Project metadata and configuration
├── assets/
│   ├── data_asset/
│   │   └── predictive_maintenance.csv              # Training dataset (10,000+ records)
│   ├── environment/                                # Environment configurations
│   ├── notebook/
│   │   └── autoai_notebook_*.ipynb                 # AutoAI generated notebook for incremental learning
│   └── wml_model/                                  # Trained machine learning models
│       ├── model_1/                                # Primary model artifacts
│       └── model_2/                                # Alternative model version
└── assettypes/
    ├── auto_ml.json                                # AutoML asset type definition
    └── wx_prompt.json                              # Watson AI prompt configurations
```

## 📊 Dataset Information

The project uses a comprehensive predictive maintenance dataset with the following features:

### Input Features
- **UDI**: Unique Data Identifier
- **Product ID**: Unique product identifier
- **Type**: Product quality variant (L, M, H for Low, Medium, High)
- **Air temperature [K]**: Ambient air temperature in Kelvin
- **Process temperature [K]**: Process temperature in Kelvin  
- **Rotational speed [rpm]**: Equipment rotational speed
- **Torque [Nm]**: Torque measurements in Newton-meters
- **Tool wear [min]**: Tool wear time in minutes

### Target Variables
- **Target**: Binary failure indicator (0 = No Failure, 1 = Failure)
- **Failure Type**: Specific failure category
  - No Failure
  - Heat Dissipation Failure
  - Power Failure
  - Overstrain Failure
  - Tool Wear Failure
  - Random Failure

### Dataset Statistics
- **Total Records**: 10,000+ equipment observations
- **Features**: 8 input features + 2 target variables
- **Data Quality**: Clean, structured industrial sensor data
- **Time Period**: Historical equipment operation data

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- IBM Watson Studio account
- IBM Cloud account (for deployment)
- Required Python packages (see requirements.txt):
  - streamlit 1.28+
  - pandas 1.5+
  - numpy 1.21+
  - plotly 5.15+
  - scikit-learn 1.3+
  - ibm-watson-machine-learning 1.0+
  - requests 2.31+

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MoB-369/predictive-maintenance.git
   cd predictive-maintenance
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Credentials**
   - Update `.streamlit/secrets.toml` with your IBM Watson ML credentials
   - See `STREAMLIT_SETUP.md` for detailed configuration instructions

4. **Run the Streamlit App**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Alternative: Jupyter Notebook**
   - Open the project in IBM Watson Studio
   - Use the AutoAI notebook for model training and experimentation

## 🖥️ Streamlit Web Application

The project includes a comprehensive Streamlit dashboard with the following features:

### 📊 Dashboard Pages

1. **🏠 Main Dashboard**
   - Equipment health overview
   - Key performance metrics
   - Failure type distribution
   - Recent alerts and notifications

2. **🔮 Single Prediction**
   - Individual equipment failure prediction
   - Interactive parameter input
   - Real-time risk assessment
   - Maintenance recommendations

3. **📊 Batch Analysis**
   - Upload CSV files for bulk predictions
   - Process multiple equipment records
   - Risk level categorization
   - Downloadable results

4. **📈 Historical Trends**
   - Time series analysis
   - Parameter correlation analysis
   - Failure rate trends
   - Equipment performance patterns

5. **⚙️ Model Information**
   - API configuration status
   - Model performance metrics
   - Feature importance analysis
   - Deployment details

### 🔌 API Integration

The Streamlit app integrates with your deployed IBM Watson ML model via REST API:

- **Real-time Predictions**: Direct API calls to your deployed model
- **Batch Processing**: Efficient bulk prediction capabilities
- **Fallback Mode**: Mock predictions when API is unavailable
- **Error Handling**: Robust error handling and user feedback
- **Security**: IBM Cloud IAM authentication

## 🔬 Model Development Process

### AutoAI Pipeline

The project uses IBM Watson Studio's AutoAI for automated machine learning:

1. **Data Preprocessing**: Automated feature engineering and data cleaning
2. **Algorithm Selection**: AutoAI tests multiple algorithms and selects the best performers
3. **Hyperparameter Optimization**: Automated tuning of model parameters
4. **Model Evaluation**: Comprehensive performance assessment
5. **Pipeline Generation**: Creation of deployable model pipeline

### Incremental Learning Workflow

The notebook supports continuous model improvement:

1. **Pipeline Retrieval**: Load existing trained model
2. **Data Batching**: Process new training data in manageable batches  
3. **Incremental Training**: Update model with `partial_fit` methodology
4. **Model Evaluation**: Test updated model performance
5. **Deployment**: Deploy improved model to production

## 📈 Model Performance

The predictive maintenance models are optimized for:

- **Precision**: Minimize false positive predictions
- **Recall**: Capture actual failure cases effectively  
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall classification performance
- **Business Metrics**: Maintenance cost reduction and uptime improvement

## 🔧 Usage

### Training New Models

1. Open the AutoAI notebook in Watson Studio
2. Configure data source and target variables
3. Run AutoAI experiment to generate optimal pipeline
4. Review model performance metrics
5. Select best performing model for deployment

### Incremental Learning

1. Prepare new training data in the same format
2. Use the incremental learning notebook
3. Load existing model pipeline
4. Apply `partial_fit` with new data batches
5. Evaluate and deploy updated model

### Making Predictions

```python
# Example prediction workflow
import pandas as pd
from ibm_watson_machine_learning import APIClient

# Load new equipment data
new_data = pd.read_csv('new_equipment_data.csv')

# Make predictions
predictions = model.predict(new_data)
failure_probabilities = model.predict_proba(new_data)
```

## 🚀 Deployment

### Local Development
- Use Jupyter notebooks for experimentation
- Test model performance with validation data
- Iterate on feature engineering and model selection

### Production Deployment
- Deploy models to IBM Watson Machine Learning
- Set up real-time scoring endpoints
- Configure batch prediction workflows
- Implement monitoring and alerting

## 🔍 Monitoring and Maintenance

- **Model Drift Detection**: Monitor for changes in data patterns
- **Performance Tracking**: Continuous evaluation of prediction accuracy
- **Retraining Schedule**: Regular model updates with new data
- **A/B Testing**: Compare model versions in production
- **Business Impact Metrics**: Track maintenance cost savings and uptime improvements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IBM Watson Studio for AutoAI capabilities
- Industrial IoT community for dataset and domain expertise
- Contributors to open-source machine learning libraries

