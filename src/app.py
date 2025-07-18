import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- 1. Data Loading and Model Training (Cached for efficiency) ---
@st.cache_resource # Use st.cache_resource for models/pipelines that are expensive to create
def load_data_and_train_model():
    """
    Loads the dataset, preprocesses it, and trains the RandomForestClassifier model.
    This function is cached by Streamlit to run only once.
    """
    print("--- Initializing Streamlit App: Loading Data and Training Model ---")
    try:
        df = pd.read_csv('adult-census-income.csv')
    except FileNotFoundError:
        df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/predicting-your-future-with-data/main/adult-census-income.csv')

    st.write("Dataset loaded successfully.")

    # Clean Column Names: Replace '.' with '_' and strip spaces
    original_columns = df.columns.tolist()
    df.columns = [col.strip().replace('.', '_') for col in original_columns]

    # Strip spaces from all object columns and replace '?' with NaN
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.replace('?', np.nan)

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Separate features (X) and target (y)
    X = df.drop('income', axis=1)
    y = df['income'].apply(lambda x: 1 if x == '>50K' else 0) # 1 for >50K, 0 for <=50K

    # Store all feature columns for consistent DataFrame creation later
    all_feature_columns = X.columns.tolist()

    # Create a default user profile template from the first row of the cleaned data
    if not df.empty:
        default_user_profile_template = df.iloc[0].drop('income', errors='ignore').to_dict()
    else:
        # Fallback if df is empty after cleaning
        default_user_profile_template = {
            'age': 30, 'workclass': 'Private', 'fnlwgt': 200000,
            'education': 'HS-grad', 'education_num': 9, 'marital_status': 'Never-married',
            'occupation': 'Other-service', 'relationship': 'Not-in-family',
            'race': 'White', 'sex': 'Male', 'capital_gain': 0, 'capital_loss': 0,
            'hours_per_week': 40, 'native_country': 'United-States'
        }

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # Store unique values for categorical features for Streamlit selectboxes
    categorical_feature_options = {}
    for col in categorical_features:
        categorical_feature_options[col] = sorted(df[col].unique().tolist())

    # Define preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Build and Train the Supervised Classification Model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])

    st.write("Training the RandomForestClassifier model...")
    model.fit(X, y) # Train on full cleaned data
    st.write("Model training complete. App is ready!")

    return model, all_feature_columns, categorical_feature_options, default_user_profile_template

# --- 2. Recommendation Logic ---
def get_recommendations(user_profile_df, model, top_n=3, min_prob_increase=0.05):
    """
    Generates recommendations for a user based on potential changes to their profile,
    aiming to increase the probability of earning >50K.

    Args:
        user_profile_df (pd.DataFrame): A DataFrame with a single row representing the user's current profile.
                                        It must have all original feature columns.
        model: The trained scikit-learn pipeline model (with preprocessor and classifier).
        top_n (int): Number of top recommendations to provide.
        min_prob_increase (float): Minimum probability increase required for a recommendation to be considered.

    Returns:
        list: A list of dictionaries, each representing a recommended change and its predicted impact.
    """
    initial_prediction_prob = model.predict_proba(user_profile_df)[:, 1][0]
    st.write(f"Your initial predicted probability of earning >50K: **{initial_prediction_prob:.2f}**")

    recommendations = []
    base_profile = user_profile_df.iloc[0].copy()

    # Define mutable features and their possible 'improved' values.
    mutable_features_options = {
        'education': ['Bachelors', 'Masters', 'Doctorate', 'Prof-school', 'Assoc-voc', 'Assoc-acdm'],
        'occupation': ['Exec-managerial', 'Prof-specialty', 'Sales', 'Tech-support'],
        'hours_per_week': [40, 50, 60],
        'workclass': ['Self-emp-inc', 'Federal-gov', 'State-gov']
    }

    for feature, potential_values in mutable_features_options.items():
        original_value = base_profile[feature]

        if feature == 'hours_per_week':
            current_hours = original_value
            values_to_try = [v for v in potential_values if v > current_hours]
        else:
            values_to_try = [v for v in potential_values if v != original_value]

        for value in values_to_try:
            temp_profile = base_profile.copy()
            temp_profile[feature] = value
            temp_profile_df = pd.DataFrame([temp_profile])

            new_prediction_prob = model.predict_proba(temp_profile_df)[:, 1][0]
            prob_increase = new_prediction_prob - initial_prediction_prob

            if prob_increase > min_prob_increase:
                recommendations.append({
                    'feature': feature,
                    'original_value': original_value,
                    'recommended_value': value,
                    'predicted_prob_increase': prob_increase,
                    'new_predicted_prob': new_prediction_prob
                })

    recommendations.sort(key=lambda x: x['predicted_prob_increase'], reverse=True)

    return recommendations[:top_n]

# --- Streamlit UI ---
st.set_page_config(page_title="Income Recommendation System", layout="centered")

st.title("🌱 Your Future with Data: Income Recommendations")
st.markdown("""
This application predicts whether a person will earn more or less than $50,000 per year based on demographic and socioeconomic data.
Based on the prediction, it suggests strategies or changes to increase the likelihood of surpassing that income threshold.
""")

# Load model and data components
with st.spinner('Loading data and training model... This might take a moment.'):
    model, all_feature_columns, categorical_feature_options, default_user_profile_template = load_data_and_train_model()

st.subheader("Enter Your Socioeconomic Profile")

# Create input widgets for user data
user_inputs = {}

# Numerical inputs
user_inputs['age'] = st.number_input(
    "Age",
    min_value=17, max_value=90,
    value=int(default_user_profile_template.get('age', 30)),
    step=1
)
user_inputs['fnlwgt'] = st.number_input(
    "Fnlwgt (Final Weight)",
    min_value=10000, max_value=1500000,
    value=int(default_user_profile_template.get('fnlwgt', 200000)),
    step=1000
)
user_inputs['education_num'] = st.number_input(
    "Education Number",
    min_value=1, max_value=16,
    value=int(default_user_profile_template.get('education_num', 9)),
    step=1,
    help="Numerical representation of education level (e.g., 9 for HS-grad, 13 for Bachelors)"
)
user_inputs['capital_gain'] = st.number_input(
    "Capital Gain",
    min_value=0, max_value=100000,
    value=int(default_user_profile_template.get('capital_gain', 0)),
    step=100
)
user_inputs['capital_loss'] = st.number_input(
    "Capital Loss",
    min_value=0, max_value=5000,
    value=int(default_user_profile_template.get('capital_loss', 0)),
    step=100
)
user_inputs['hours_per_week'] = st.number_input(
    "Hours per Week",
    min_value=1, max_value=99,
    value=int(default_user_profile_template.get('hours_per_week', 40)),
    step=1
)

# Categorical inputs (using selectbox with options from loaded data)
for col, options in categorical_feature_options.items():
    if col not in user_inputs: # Avoid re-creating inputs for numerical features
        default_index = 0
        if default_user_profile_template.get(col) in options:
            default_index = options.index(default_user_profile_template.get(col))
        user_inputs[col] = st.selectbox(
            col.replace('_', ' ').title(), # Make label more readable
            options,
            index=default_index
        )

# Button to trigger recommendations
if st.button("Get Recommendations"):
    # Create DataFrame from user inputs, ensuring correct column order
    user_profile_df = pd.DataFrame([user_inputs], columns=all_feature_columns)

    # Display recommendations
    st.subheader("Your Personalized Recommendations:")
    try:
        recommendations = get_recommendations(user_profile_df, model)

        if recommendations:
            for i, rec in enumerate(recommendations):
                st.markdown(f"""
                <div style="background-color: #e0f2f7; border-left: 5px solid #007bb5; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                    <p style="font-size: 1.1em; font-weight: bold;">{i+1}. Change {rec['feature'].replace('_', ' ')} from '{rec['original_value']}' to '<strong>{rec['recommended_value']}</strong>'.</p>
                    <p><i>(Predicted probability of >50K would increase by {rec['predicted_prob_increase']:.2f} to {rec['new_predicted_prob']:.2f})</i></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant recommendations found for this profile. Your current profile might already have a high predicted income likelihood, or simple changes don't yield substantial improvements.")

    except Exception as e:
        st.error(f"An error occurred while generating recommendations: {e}")



    ### to run the program please type $ streamlit run src/app.py ###