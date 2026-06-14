import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import plotly.express as px

st.set_page_config(
    page_title="NayePankh Beneficiary Recommendation System",
    page_icon="🎯",
    layout="wide")

st.title("NayePankh Beneficiary Assistance Recommendation System")

def load_data():
    return pd.read_csv("beneficiary_data.csv")

df = load_data()

le_gender = LabelEncoder()
le_education = LabelEncoder()
le_location = LabelEncoder()
le_disability = LabelEncoder()
le_program = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Education_Level"] = le_education.fit_transform(df["Education_Level"])
df["Location"] = le_location.fit_transform(df["Location"])
df["Disability"] = le_disability.fit_transform(df["Disability"])
df["Program"] = le_program.fit_transform(df["Program"])


X = df.drop("Program", axis=1)
y = df["Program"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = RandomForestClassifier( n_estimators=200, random_state=42)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

st.sidebar.header("Navigation")

menu = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Recommendation System",
        "Model Performance"
    ])

if menu == "Dashboard":

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Beneficiaries",len(df))

    col2.metric( "Programs",len(le_program.classes_) )

    col3.metric("Model Accuracy",f"{accuracy*100:.2f}%")

    st.subheader("Sample Data")

    st.dataframe(df.head())

    st.subheader("Program Distribution")

    program_names = le_program.inverse_transform(df["Program"])

    chart_df = pd.DataFrame( {"Program": program_names})

    fig = px.histogram(
        chart_df,
        x="Program",
        title="Beneficiary Distribution by Program")

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Income Distribution")

    income_fig = px.histogram(df, x="Monthly_Income",nbins=20,title="Monthly Income Distribution")

    st.plotly_chart(income_fig,use_container_width=True )

elif menu == "Recommendation System":

    st.header("Get Assistance Recommendation")

    age = st.number_input("Age",min_value=10, max_value=100, value=20)

    gender = st.selectbox("Gender", ["Male", "Female"])

    income = st.number_input( "Monthly Income (₹)", min_value=0,value=3000)

    education = st.selectbox("Education Level",["School", "Graduate" , "None" ])

    family_size = st.number_input("Family Size",min_value=1, max_value=15,value=5)

    location = st.selectbox("Location",["Rural", "Urban"])

    disability = st.selectbox("Disability",["No", "Yes"])

    if st.button("Recommend Program"):

        input_data = np.array([[
            age,
            le_gender.transform([gender])[0],
            income,
            le_education.transform([education])[0],
            family_size,
            le_location.transform([location])[0],
            le_disability.transform([disability])[0]
        ]])

        prediction = model.predict(input_data)

        recommendation = ( le_program.inverse_transform(prediction)[0] )

        probabilities = model.predict_proba(input_data)[0]

        top_indices = np.argsort(probabilities )[::-1][:3]

        st.success( f"Recommended Program: {recommendation}" )

        st.subheader("Top 3 Recommended Programs")

        results = []

        for idx in top_indices:
            results.append({
                "Program":
                le_program.inverse_transform([idx])[0],

                "Confidence (%)":
                round(probabilities[idx]*100,2)
            })

        st.table(pd.DataFrame(results))




elif menu == "Model Performance":

    st.header("Model Information")

    st.metric("Random Forest Accuracy",f"{accuracy*100:.2f}%" )

    importance = model.feature_importances_

    feature_df = pd.DataFrame({"Feature": X.columns, "Importance": importance} )

    feature_df = feature_df.sort_values(by="Importance",ascending=False)

    fig = px.bar(
        feature_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance")

    st.plotly_chart(fig,use_container_width=True)

    st.dataframe(feature_df)
