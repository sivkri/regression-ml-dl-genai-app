# regression-ml-dl-genai-app

An interactive regression analysis platform powered by **Machine Learning**, **Deep Learning**, and **Generative AI** — built with **Streamlit** for real-time, explainable model exploration.

---

## Project Summary

This project demonstrates a complete workflow for regression modeling using both **traditional ML algorithms** and **Deep Learning architectures**, enhanced by **Generative AI (OpenAI GPT)** to explain predictions and metrics in human-friendly language.

Designed as a modular, production-style app, this solution enables users to:
- Upload and preprocess their own data
- Select and compare regression models
- Visualize model performance
- Understand predictions with the help of a GenAI assistant

---

## 🎯 Key Features

| Feature | Description |
|--------|-------------|
| **ML & DL Models** | Train & evaluate Linear Regression, Random Forest, and Deep Neural Networks |
| **Model Evaluation** | Automatically computes metrics like RMSE, MAE, R², and plots |
| **GenAI Explanation** | Uses OpenAI GPT to generate plain-language descriptions of the models |
| **Interactive UI** | Built with Streamlit for easy interaction and dynamic insights |
| **Data Upload** | Accepts CSV data
---

## Technologies Used

- **Scikit-learn**: Linear Regression, Random Forest
- **TensorFlow / Keras**: Deep Learning regression model
- **Pandas & Matplotlib**: Data manipulation and visualization
- **Streamlit**: Interactive web app
- **OpenAI GPT (via API)**: Model explanation using Generative AI

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sivkri/regression-ml-dl-genai-app.git
cd regression-ml-dl-genai-app
```
### 2. Install the requirements

```bash
pip install -r requirements.txt
```
### 3. Get your OpenAI API key

Go to OpenAI API keys

you can input the key securely via the app if prompted.

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```
## How It Works

**Upload CSV Data**: Choose a CSV file with numeric features and a continuous target.

**Choose Model**: Select between ML (Linear Regression, Random Forest) or DL (Keras DNN).

**Model Training**: Train and evaluate the model.

**Visualization**: Plot actual vs predicted values, feature importance, etc.

**Explain with GenAI**: Generate human-friendly summaries and insights using GPT.

## Sample Outputs

- Scatter plots of predictions

- Evaluation metrics (MAE, RMSE, R²)

- GPT-generated explanations of model performance

- Highlighted prediction confidence and anomalies

## Generative AI Integration
The GenAI assistant uses the OpenAI API (GPT-3.5/4) to generate:

  Explanation of model performance

  Suggestions on data preprocessing

  Interpretations of model predictions in natural language

  
