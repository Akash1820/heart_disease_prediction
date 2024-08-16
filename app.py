import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import gdown


@st.cache_data

def load_data():
    try:
    #    Load and preprocess the data
        file_id = '1Pl9q58dqA8dV_JEOby8TPBw2GHZN1bhz'  # Replace with your file ID

# Generate the direct download URL
        url = f'https://drive.google.com/uc?id={file_id}'

# Load the file directly into a pandas DataFrame
        heart = pd.read_csv(gdown.download(url, quiet=False))

        # heart = pd.read_csv("C:\\Users\\akash\\OneDrive - Lambton College\\project\\big data viz project 2\\archive\\2022\\heart_2022_with_nans.csv")
        heart.drop_duplicates(inplace=True)

    # Handle missing values
        for col in heart.columns:
            if heart[col].dtype == 'object':
                min_value = heart[col].dropna().mode().iloc[0]
                heart[col] = heart[col].fillna(min_value)
            else:
                mean_value = heart[col].mean()
                heart[col] = heart[col].fillna(mean_value)

    # Encode categorical variables
        for col in heart.select_dtypes(include='object').columns:
            encoder = LabelEncoder()
            heart[col] = encoder.fit_transform(heart[col])

        return heart
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(heart):
    try:
    # Select features based on correlation
        correlation_matrix = heart.corr()
        strong_correlation_cols = [col for col in correlation_matrix.columns if abs(correlation_matrix.loc['HadHeartAttack', col]) >= 0.06 and col != 'HadHeartAttack' and col != 'AgeCategory' and col != 'LastCheckupTime']

        X = heart[strong_correlation_cols]
        y = heart['HadHeartAttack']

    # Split the data and apply undersampling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        under_sampler = RandomUnderSampler(random_state=42)
        X_train, y_train = under_sampler.fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test, sc
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None, None, None, None, None

heart = load_data()
if heart is not None:
    X_train, X_test, y_train, y_test, sc = preprocess_data(heart)

    if X_train is not None:


    # Initialize the models
        models = {
            'KNN': KNeighborsClassifier(),
            'RandomForest': RandomForestClassifier(),
            'DecisionTree': DecisionTreeClassifier(),
            'LogisticRegression': LogisticRegression(),
            'SVC': SVC()
        }   
        # Model selection
        st.title("Heart Disease Prediction App")
        st.write("Enter the details to predict the risk of heart disease.")
        model_name = st.selectbox("Choose the Model", ["KNN", "RandomForest", "DecisionTree", "LogisticRegression", "SVC"])



        # Train the selected model
        mod = models[model_name]
        mod.fit(X_train, y_train)
        y_pred = mod.predict(X_test)

        # Evaluate the model
        test_accuracy = accuracy_score(y_test, y_pred)

        # Feature inputs by user
        st.write("### Select the feature")
       
        # Binary inputs with 'Yes'/'No' options
        def yes_no_to_binary(value):
            return 1 if value == 'Yes' else 0

        sex = yes_no_to_binary(st.radio("Sex", ('Male', 'Female')))
        PhysicalHealthDays = yes_no_to_binary(st.radio("Physical Health Days", ('Yes', 'No')))
        PhysicalActivities = yes_no_to_binary(st.radio("Physical Activities", ('Yes', 'No')))
        RemovedTeeth = yes_no_to_binary(st.radio("Removed Teeth", ('Yes', 'No')))
        HadAngina = yes_no_to_binary(st.radio("Had Angina", ('Yes', 'No')))
        HadStroke = yes_no_to_binary(st.radio("Had Stroke", ('Yes', 'No')))
        HadCOPD = yes_no_to_binary(st.radio("Had COPD", ('Yes', 'No')))
        HadKidneyDisease = yes_no_to_binary(st.radio("Had Kidney Disease", ('Yes', 'No')))
        HadArthritis = yes_no_to_binary(st.radio("Had Arthritis", ('Yes', 'No')))
        HadDiabetes = yes_no_to_binary(st.radio("Had Diabetes", ('Yes', 'No')))
        DeafOrHardOfHearing = yes_no_to_binary(st.radio("Deaf or Hard of Hearing", ('Yes', 'No')))
        BlindOrVisionDifficulty = yes_no_to_binary(st.radio("Blind or Vision Difficulty", ('Yes', 'No')))
        DifficultyWalking = yes_no_to_binary(st.radio("Difficulty Walking", ('Yes', 'No')))
        DifficultyDressingBathing = yes_no_to_binary(st.radio("Difficulty Dressing or Bathing", ('Yes', 'No')))
        DifficultyErrands = yes_no_to_binary(st.radio("Difficulty Running Errands", ('Yes', 'No')))
        SmokerStatus = yes_no_to_binary(st.radio("Smoker Status", ('Yes', 'No')))
        ChestScan = yes_no_to_binary(st.radio("Chest Scan", ('Yes', 'No')))
        AlcoholDrinkers = yes_no_to_binary(st.radio("Alcohol Drinkers", ('Yes', 'No')))
        PneumoVaxEver = yes_no_to_binary(st.radio("Vaccinated for Pneumo Ever", ('Yes', 'No')))


        # Collect all inputs into a list or dictionary
        user_input = {
            'Sex': sex,
            'PhysicalHealthDays': PhysicalHealthDays,
            'PhysicalActivities': PhysicalActivities,
            'RemovedTeeth': RemovedTeeth,
            'HadAngina': HadAngina,
            'HadStroke': HadStroke,
            'HadCOPD': HadCOPD,
            'HadKidneyDisease': HadKidneyDisease,
            'HadArthritis': HadArthritis,
            'HadDiabetes': HadDiabetes,
            'DeafOrHardOfHearing': DeafOrHardOfHearing,
            'BlindOrVisionDifficulty': BlindOrVisionDifficulty,
            'DifficultyWalking': DifficultyWalking,
            'DifficultyDressingBathing': DifficultyDressingBathing,
            'DifficultyErrands': DifficultyErrands,
            'SmokerStatus': SmokerStatus,
            'ChestScan': ChestScan,
            'AlcoholDrinkers': AlcoholDrinkers,
            'PneumoVaxEver': PneumoVaxEver
        }

        # Convert user input to DataFrame
        user_input_df = pd.DataFrame([user_input])

        # Ensure features match the trained model's expectations
        user_input_sc = sc.transform(user_input_df)

        # Prediction
        if st.button('Predict'):
            try:
                # Ensure features match the trained model's expectations
                user_input_sc = sc.transform(user_input_df)
                # Predict the risk of heart disease
                prediction = mod.predict(user_input_sc)
                # Output the prediction
                if prediction[0] == 1:
                    st.write("The model predicts a high risk of heart disease.")
                else:
                    st.write("The model predicts a low risk of heart disease.")

                # Display evaluation metrics
                st.write("### Evaluation Metrics")
                st.write("Test Accuracy:", test_accuracy)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.error("Error in preprocessing data.")
else:
    st.error("Error loading data.")
