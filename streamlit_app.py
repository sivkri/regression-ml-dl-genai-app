import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from openai import OpenAI

# App title
st.title("🔍 Regression Analysis App with ML, DL & GenAI")

# File upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.write(df.head())

    # Feature selection
    st.sidebar.header("Model Configuration")
    features = st.sidebar.multiselect("Select Feature Columns", options=df.columns.tolist(), default=df.columns[:-1].tolist())
    target = st.sidebar.selectbox("Select Target Column", options=[col for col in df.columns if col not in features])

    # Train/test split
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model choice
    model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest", "Deep Learning"])
    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_choice == "Random Forest":
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_choice == "Deep Learning":
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=100, verbose=0)
        y_pred = model.predict(X_test).flatten()

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.subheader("📊 Model Evaluation")
    st.write(f"**Mean Squared Error:** {mse:.4f}")
    st.write(f"**R-squared:** {r2:.4f}")

    # Plotting
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    # GenAI explanation
    st.subheader("🧠 Explain Prediction using GenAI")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        prompt = (
            f"Explain the regression model results:\n"
            f"MSE = {mse:.4f}, R-squared = {r2:.4f}.\n"
            f"Explain what this means in layman terms."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful data scientist."},
                {"role": "user", "content": prompt}
            ]
        )

        explanation = response.choices[0].message.content
        st.markdown("### 🤖 GenAI Explanation")
        st.write(explanation)
