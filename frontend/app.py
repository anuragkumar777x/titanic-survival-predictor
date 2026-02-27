import streamlit as st
import requests
import pandas as pd
import plotly.express as px

df = pd.read_csv("backend/titanic_cleaned.csv")

st.set_page_config(
    page_title="Titanic Survival Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- SIDEBAR ----
st.sidebar.title("🚢 Titanic App")

page = st.sidebar.radio(
    "Navigate",
    ["Prediction", "Survival Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app predicts whether a passenger would have survived the Titanic disaster "
    "using a Logistic Regression model trained on historical data."
)

if page == "Prediction":

    st.title("🚢 Titanic Survival Predictor")

    st.markdown("Enter your details below to see if you would have survived the Titanic disaster.")

    with st.form("prediction_form"):

        age = st.slider("Age", 1, 80, 25)

        pclass = st.selectbox("Passenger Class", [1, 2, 3])

        sex = st.selectbox("Sex", ["male", "female"])

        embarked_option = st.selectbox(
            "Embarked Port",
            ["Southampton", "Queenstown", "Cherbourg"]
        )

        # Map full names to backend codes
        embarked_mapping = {
            "Southampton": "S",
            "Queenstown": "Q",
            "Cherbourg": "C"
        }

        embarked = embarked_mapping[embarked_option]

        family_type = st.selectbox("Family Type", ["alone", "medium", "large"])

        submitted = st.form_submit_button("Predict")

    if submitted:

        data = {
            "Age": age,
            "Pclass": pclass,
            "Sex": sex,
            "Embarked": embarked,
            "FamilyType": family_type
        }

        try:
            response = requests.post(
                "https://titanic-survival-predictor-27ix.onrender.com",
                json=data
            )

            result = response.json()

            prediction = result["prediction"]
            probability = result["survival_probability"]

            if prediction == 1:
                st.success(f"🎉 You would have survived! You likely got a lifeboat ({probability*100:.2f}% chance)")
            else:
                st.error(f"Unfortunately, you wouldn't have made it. RIP 🕯️ ({probability*100:.2f}% chance)")

        except Exception as e:
            st.error("Backend is not running or connection failed.")

elif page == "Survival Analysis":

    st.title("📊 Titanic Survival Analysis")

    st.markdown("Select the analysis you want to explore:")

    analysis_option = st.selectbox(
        "Choose Analysis",
        [
            "Survival Distribution",
            "Survival Rate by Sex",
            "Survival Rate by Passenger Class",
            "Age Distribution"
        ]
    )

    st.markdown("---")

    if analysis_option == "Survival Distribution":
        st.subheader("Overall Survival Distribution")

        survival_counts = df["Survived"].value_counts().reset_index()
        survival_counts.columns = ["Survived", "Count"]
        survival_counts["Survived"] = survival_counts["Survived"].map({0: "Did Not Survive", 1: "Survived"})

        fig = px.pie(
            survival_counts,
            names="Survived",
            values="Count",
            color="Survived",
            color_discrete_map={
                "Survived": "green",
                "Did Not Survive": "red"
            },
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_option == "Survival Rate by Sex":
        st.subheader("Survival Rate by Sex")

        sex_survival = df.groupby("Sex")["Survived"].mean().reset_index()

        fig = px.bar(
            sex_survival,
            x="Sex",
            y="Survived",
            color="Sex",
            color_discrete_sequence=["#1f77b4", "#ff69b4"]
        )

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_option == "Survival Rate by Passenger Class":
        st.subheader("Survival Rate by Passenger Class")

        class_survival = df.groupby("Pclass")["Survived"].mean().reset_index()

        fig = px.bar(
            class_survival,
            x="Pclass",
            y="Survived",
            color="Pclass",
            color_discrete_sequence=["#2ca02c", "#ff7f0e", "#d62728"]
        )

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_option == "Age Distribution":
        st.subheader("Age Distribution")

        fig = px.histogram(
            df,
            x="Age",
            nbins=30,
            color_discrete_sequence=["#636EFA"]
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Model Performance")
    st.write("Accuracy: ~82%")
    st.write("Model: Logistic Regression")
    st.write("Features Used: Age, Pclass, Sex, Embarked, Family Type")
