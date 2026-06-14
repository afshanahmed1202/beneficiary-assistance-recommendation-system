# Beneficiary Assistance Recommendation System

## Overview

The Beneficiary Assistance Recommendation System is a Machine Learning-based application developed to support NGOs such as NayePankh Foundation in identifying and recommending the most suitable assistance programs for beneficiaries.

The system analyzes beneficiary information such as age, income, education level, family size, location, and disability status to predict the most appropriate support program. This helps NGOs allocate resources efficiently and ensure that assistance reaches the people who need it most.

---

## Problem Statement

NGOs often manage multiple welfare programs, including:

* Education Support
* Scholarship Programs
* Food Distribution
* Skill Development
* Healthcare Assistance
* Women Empowerment

Manually determining which program best fits a beneficiary can be time-consuming and inconsistent. This project uses Machine Learning to automate the recommendation process and support data-driven decision-making.

---

## Features

* Beneficiary profile analysis
* Assistance program recommendation
* Top 3 recommended programs with confidence scores
* Interactive Streamlit dashboard
* Data visualization and analytics
* Feature importance analysis
* Machine Learning model evaluation

---

## Dataset

The dataset contains beneficiary information with the following features:

| Feature         | Description               |
| --------------- | ------------------------- |
| Age             | Beneficiary age           |
| Gender          | Male/Female               |
| Monthly_Income  | Monthly family income     |
| Education_Level | None, School, Graduate    |
| Family_Size     | Number of family members  |
| Location        | Rural or Urban            |
| Disability      | Yes or No                 |
| Program         | Target assistance program |

Target Variable:

```text
Program
```

Possible Programs:

* Education Support
* Scholarship Program
* Food Distribution
* Skill Development
* Healthcare Assistance
* Women Empowerment

---

## Machine Learning Workflow

### 1. Data Collection

Beneficiary information is collected and stored in a CSV dataset.

### 2. Data Preprocessing

* Handling categorical variables
* Label Encoding
* Feature selection

### 3. Model Training

The project uses a Random Forest Classifier to predict the most suitable assistance program.

### 4. Evaluation

Model performance is evaluated using:

* Accuracy Score
* Feature Importance Analysis

### 5. Recommendation

The system generates:

* Recommended Program
* Top 3 Recommendations
* Confidence Scores

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Plotly
* Matplotlib
* Seaborn

### Machine Learning Algorithm

* Random Forest Classifier

---

## Project Structure

```text
Beneficiary_Assistance_Recommendation_System/
│
├── app.py
├── beneficiary_data.csv
├── requirements.txt
├── README.md
│
└── screenshots/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/beneficiary-assistance-recommendation-system.git
```

### Navigate to Project Folder

```bash
cd beneficiary-assistance-recommendation-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will start at:

```text
http://localhost:8501
```

---

## Dashboard Features

### Dataset Overview

* Total Beneficiaries
* Available Programs
* Model Accuracy

### Visualizations

* Program Distribution
* Income Distribution
* Feature Importance

### Recommendation System

Input beneficiary details and receive:

* Recommended Program
* Top 3 Program Suggestions
* Confidence Scores

---

## Future Enhancements

* Integration with real NGO databases
* Beneficiary priority scoring
* Geographic impact analysis
* Deep Learning-based recommendations
* Cloud deployment
* Admin dashboard
* PDF report generation
* Multi-language support

---

## Impact

This project demonstrates how Machine Learning can support social welfare organizations by improving beneficiary identification, optimizing resource allocation, and enabling data-driven decision-making.

The system can help NGOs increase efficiency, improve transparency, and maximize the impact of their social programs.

---

## Author

Developed as a Machine Learning project for NGO-focused beneficiary assistance and recommendation systems.
