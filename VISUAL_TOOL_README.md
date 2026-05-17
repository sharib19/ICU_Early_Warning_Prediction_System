# 🏥 ICU Early Warning Prediction System - Visual Tool

A complete interactive Streamlit dashboard for your Kaggle ICU prediction model. Run and visualize your data without any modifications to the original notebook!

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sharib19/ICU_Early_Warning_Prediction_System.git
cd ICU_Early_Warning_Prediction_System

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## 📊 Features

### 1️⃣ **Dashboard View** 📊
- Real-time patient statistics
- Vital signs distribution charts
- Patient risk assessment overview
- Live data table preview

### 2️⃣ **Model Training View** 🔬
- Train XGBoost and Random Forest models
- Adjust hyperparameters in real-time
- Compare model performance
- View ROC curves and confusion matrices
- Feature importance analysis

### 3️⃣ **Predictions View** 📈
- Enter patient vital signs interactively
- Get real-time risk assessment
- Automatic clinical alert system
- Risk score visualization
- Alerts for abnormal vitals

### 4️⃣ **About View** 📋
- Project overview
- Clinical applications
- Model descriptions
- Performance metrics

---

## 🎯 Input Vital Signs

The system monitors:
- **Heart Rate**: 40-160 bpm
- **Systolic BP**: 80-180 mmHg
- **Diastolic BP**: 40-120 mmHg
- **Respiratory Rate**: 8-40 bpm
- **Temperature**: 35-41°C
- **Oxygen Saturation**: 80-100%
- **Glucose Level**: 70-300 mg/dL
- **MEWS Score**: 0-3

---

## 🔬 Models Included

### XGBoost Classifier
- Gradient boosting framework
- Configurable max depth (3-10)
- Excellent for binary classification
- Fast training and inference

### Random Forest Classifier
- Ensemble learning method
- Configurable number of estimators (10-200)
- Interpretable predictions
- Robust feature importance

---

## 📈 Performance Metrics

- **AUC-ROC Score**: Classification discrimination
- **Confusion Matrix**: True/False positives and negatives
- **ROC Curves**: Model comparison
- **Feature Importance**: Which vitals matter most

---

## 🔗 Integration with Your Kaggle Notebook

This tool works **alongside** your Kaggle notebook:

1. Your original notebook remains **completely unchanged** ✅
2. The visual tool uses **generated synthetic data** for demo
3. To use your **actual Kaggle data**, modify the `load_icu_data()` function:

```python
@st.cache_data
def load_icu_data():
    # Replace with your Kaggle dataset loading
    df = pd.read_csv('your_kaggle_dataset.csv')
    return df
```

---

## 💡 Clinical Alerts

The system automatically flags:
- ⚠️ Tachycardia (HR > 110)
- ⚠️ Bradycardia (HR < 60)
- ⚠️ Tachypnea (RR > 25)
- ⚠️ Low Oxygen (O₂ < 92%)
- ⚠️ Fever (Temp > 38.5°C)
- ⚠️ Hypothermia (Temp < 36°C)
- ⚠️ Elevated MEWS Score

---

## 🎨 UI Features

- **Professional Healthcare Color Scheme**
- **Interactive Plotly Charts**
- **Real-time Sliders** for vital signs
- **Status Indicators** with color coding
- **Responsive Sidebar Navigation**
- **Mobile-Friendly Layout**

---

## 📁 File Structure

```
ICU_Early_Warning_Prediction_System/
├── streamlit_app.py              # Main visual tool
├── requirements.txt              # Python dependencies
├── icu-patient-monitoring-and-prediction.ipynb  # Original Kaggle notebook
├── README.md                     # Project description
└── VISUAL_TOOL_README.md        # This file
```

---

## 🌐 Deployment Options

### Deploy to Streamlit Cloud (Free)
1. Push to GitHub ✅ (Already done!)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect GitHub repo
4. Select `streamlit_app.py` as main file
5. Deploy!

### Deploy to Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py" > Procfile

# Deploy
heroku login
heroku create your-app-name
git push heroku main
```

---

## 🛠️ Customization

### Change Data Source
Edit `load_icu_data()` function to load from:
- CSV files
- Kaggle API
- Database connections
- Real-time streaming

### Adjust Model Hyperparameters
Modify sliders in Model Training view or edit code:
```python
xgb_depth = st.slider("XGBoost Max Depth", 3, 10, 6)
rf_estimators = st.slider("Random Forest Estimators", 10, 200, 100)
```

### Add More Features
- Patient demographics
- Medical history
- Lab results
- Medication data

---

## 🚨 Limitations & Disclaimers

⚠️ **For Educational/Research Use Only**
- This tool is not FDA-approved
- Should NOT be used for actual clinical decisions
- Always involve qualified medical professionals
- Synthetic data is used for demo purposes

---

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [MEWS Scoring System](https://en.wikipedia.org/wiki/Modified_Early_Warning_Score)

---

## 📧 Support

For issues or questions:
1. Check the original Kaggle notebook
2. Review Streamlit documentation
3. Post on GitHub Issues
4. Contact: sharibj966@gmail.com

---

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Predicting! 🏥💊🚀**
