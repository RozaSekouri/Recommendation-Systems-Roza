import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Global Variables for Model and Preprocessor ---
# These will be loaded/trained once when the app starts
model = None
all_feature_columns = None # To store the order of columns for consistent DataFrame creation

# --- Data Loading and Preprocessing (Executed once on app startup) ---
def load_and_preprocess_data():
    global model, all_feature_columns

    print("--- Initializing Flask App: Loading Data and Training Model ---")
    try:
        df = pd.read_csv('adult-census-income.csv')
    except FileNotFoundError:
        df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/predicting-your-future-with-data/main/adult-census-income.csv')

    print("Dataset loaded successfully.")

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

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

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

    # Split data into training and testing sets (though only model training is needed for app)
    # We train on the full dataset available after cleaning for the deployed model
    # For robust evaluation, you'd use X_train, y_train, X_test, y_test
    # Here, we'll train on the entire cleaned dataset for the production model
    X_train_full = X
    y_train_full = y

    print("Data preprocessing complete for model training.")

    # Build and Train the Supervised Classification Model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])

    print("Training the RandomForestClassifier model...")
    model.fit(X_train_full, y_train_full)
    print("Model training complete. App is ready!")

# Call the function to load data and train model when the app starts
load_and_preprocess_data()

# --- Recommendation Logic (from previous step) ---
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
    print(f"User's initial predicted probability of earning >50K: {initial_prediction_prob:.2f}")

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

# --- Flask Routes ---

@app.route('/')
def home():
    """Simple home route to confirm the app is running."""
    return "Welcome to the Income Recommendation API! Use /recommend to get recommendations."

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint to receive a user profile and return income recommendations.
    Expects a JSON body with user data.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    user_data = request.get_json()
    print(f"Received user data: {user_data}")

    # Create a base profile template with default values for all features
    # This ensures consistency for the ColumnTransformer
    # You might want to get these defaults from the most frequent values in your training data
    # For simplicity, using a sample from the dataset here.
    default_profile = {
        'age': 30, 'workclass': 'Private', 'fnlwgt': 200000,
        'education': 'HS-grad', 'education_num': 9, 'marital_status': 'Never-married',
        'occupation': 'Other-service', 'relationship': 'Not-in-family',
        'race': 'White', 'sex': 'Male', 'capital_gain': 0, 'capital_loss': 0,
        'hours_per_week': 40, 'native_country': 'United-States'
    }

    # Update default profile with user-provided data
    for key, value in user_data.items():
        if key in default_profile:
            default_profile[key] = value
        else:
            print(f"Warning: User provided unknown key '{key}'. Ignoring.")

    # Create a DataFrame from the combined profile, ensuring column order
    # This is crucial for the preprocessor
    user_profile_df = pd.DataFrame([default_profile], columns=all_feature_columns)

    try:
        recommendations = get_recommendations(user_profile_df, model)
        return jsonify(recommendations), 200
    except Exception as e:
        print(f"Error during recommendation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    # In a production environment, you would use a more robust WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)