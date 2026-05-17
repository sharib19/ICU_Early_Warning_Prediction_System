# ICU Early Warning Prediction System

## AI-Powered Patient Deterioration Detection Using ICU Vital Signs

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-orange.svg)](https://xgboost.ai/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

## 📋 Overview

This project develops an AI-based early warning system for Intensive Care Units (ICU) that predicts patient deterioration 4-6 hours in advance using continuous vital signs monitoring. The system provides real-time risk scores and clinical recommendations to enable proactive intervention.

## 🎯 Problem Statement

ICU patients are at risk of sudden clinical deterioration. Traditional monitoring relies on threshold-based alarms that trigger only after deterioration occurs. This system predicts deterioration before it happens, allowing clinicians to intervene earlier.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Real-time Prediction** | Risk score (0-100%) updated with each vital sign reading |
| **MEWS Calculation** | Modified Early Warning Score based on vital signs |
| **Dual Models** | XGBoost and Random Forest classifiers |
| **Clinical Alerts** | Low/Moderate/High/Critical risk levels with actions |
| **Visualizations** | ROC curves, feature importance, confusion matrices |
| **Explainable AI** | Feature importance showing which vitals drive predictions |

## 🏗️ Architecture
