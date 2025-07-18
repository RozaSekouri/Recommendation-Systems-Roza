import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify, render_template_string # Import render_template_string

app = Flask(__name__)

# --- Global Variables for Model, Preprocessor, and Feature Options ---
model = None
all_feature_columns = None # To store the order of columns for consistent DataFrame creation
categorical_feature_options = {} # To store unique options for dropdowns in the HTML form
default_user_profile_template = {} # A template for a new user profile

# --- Data Loading and Preprocessing (Executed once on app startup) ---
def load_and_preprocess_data():
    global model, all_feature_columns, categorical_feature_options, default_user_profile_template

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

    # Create a default user profile template from the first row of the cleaned data
    # This ensures all columns are present and have valid types/values as a starting point
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

    # Store unique values for categorical features for the HTML form
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

    # Train on the entire cleaned dataset for the deployed model
    print("Data preprocessing complete for model training.")

    # Build and Train the Supervised Classification Model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])

    print("Training the RandomForestClassifier model...")
    model.fit(X, y) # Train on full cleaned data
    print("Model training complete. App is ready!")

# Call the function to load data and train model when the app starts
load_and_preprocess_data()

# --- Recommendation Logic ---
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
    # These values are chosen based on common sense and potential for higher income.
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
    return "Welcome to the Income Recommendation API! Use /input_form to enter data."

@app.route('/input_form')
def input_form():
    """
    Serves the HTML form for user input.
    Dynamically populates select options based on loaded data.
    """
    # HTML content for the form
    # This uses f-strings to embed Python variables directly into the HTML
    # for dynamic dropdown options and default values.
    form_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Income Recommendation Input</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Inter', sans-serif; }}
            .form-label {{ font-weight: 600; margin-bottom: 4px; display: block; }}
            .form-input, .form-select {{
                width: 100%;
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 8px;
                margin-bottom: 16px;
                box-sizing: border-box;
            }}
            .form-button {{
                background-color: #4F46E5;
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                transition: background-color 0.3s ease;
            }}
            .form-button:hover {{
                background-color: #4338CA;
            }}
            .recommendation-card {{
                background-color: #f0f9ff;
                border: 1px solid #bfdbfe;
                border-radius: 8px;
                padding: 16px;
                margin-top: 20px;
            }}
            .recommendation-item {{
                margin-bottom: 8px;
            }}
        </style>
    </head>
    <body class="bg-gray-100 flex items-center justify-center min-h-screen py-10">
        <div class="bg-white p-8 rounded-lg shadow-xl w-full max-w-md">
            <h1 class="text-2xl font-bold text-center mb-6 text-gray-800">Get Income Recommendations</h1>
            <form id="recommendationForm">
                <label for="age" class="form-label">Age:</label>
                <input type="number" id="age" name="age" class="form-input" value="{default_user_profile_template.get('age', 30)}" required>

                <label for="workclass" class="form-label">Workclass:</label>
                <select id="workclass" name="workclass" class="form-select" required>
                    {''.join([f'<option value="{opt}" {"selected" if opt == default_user_profile_template.get("workclass") else ""}>{opt}</option>' for opt in categorical_feature_options.get('workclass', [])])}
                </select>

                <label for="fnlwgt" class="form-label">Fnlwgt:</label>
                <input type="number" id="fnlwgt" name="fnlwgt" class="form-input" value="{default_user_profile_template.get('fnlwgt', 200000)}" required>

                <label for="education" class="form-label">Education:</label>
                <select id="education" name="education" class="form-select" required>
                    {''.join([f'<option value="{opt}" {"selected" if opt == default_user_profile_template.get("education") else ""}>{opt}</option>' for opt in categorical_feature_options.get('education', [])])}
                </select>

                <label for="education_num" class="form-label">Education Num:</label>
                <input type="number" id="education_num" name="education_num" class="form-input" value="{default_user_profile_template.get('education_num', 9)}" required>

                <label for="marital_status" class="form-label">Marital Status:</label>
                <select id="marital_status" name="marital_status" class="form-select" required>
                    {''.join([f'<option value="{opt}" {"selected" if opt == default_user_profile_template.get("marital_status") else ""}>{opt}</option>' for opt in categorical_feature_options.get('marital_status', [])])}
                </select>

                <label for="occupation" class="form-label">Occupation:</label>
                <select id="occupation" name="occupation" class="form-select" required>
                    {''.join([f'<option value="{opt}" {"selected" if opt == default_user_profile_template.get("occupation") else ""}>{opt}</option>' for opt in categorical_feature_options.get('occupation', [])])}
                </select>

                <label for="relationship" class="form-label">Relationship:</label>
                <select id="relationship" name="relationship" class="form-select" required>
                    {''.join([f'<option value="{opt}" {"selected" if opt == default_user_profile_template.get("relationship") else ""}>{opt}</option>' for opt in categorical_feature_options.get('relationship', [])])}
                </select>

                <label for="race" class="form-label">Race:</label>
                <select id="race" name="race" class="form-select" required>
                    {''.join([f'<option value="{opt}" {"selected" if opt == default_user_profile_template.get("race") else ""}>{opt}</option>' for opt in categorical_feature_options.get('race', [])])}
                </select>

                <label for="sex" class="form-label">Sex:</label>
                <select id="sex" name="sex" class="form-select" required>
                    {''.join([f'<option value="{opt}" {"selected" if opt == default_user_profile_template.get("sex") else ""}>{opt}</option>' for opt in categorical_feature_options.get('sex', [])])}
                </select>

                <label for="capital_gain" class="form-label">Capital Gain:</label>
                <input type="number" id="capital_gain" name="capital_gain" class="form-input" value="{default_user_profile_template.get('capital_gain', 0)}" required>

                <label for="capital_loss" class="form-label">Capital Loss:</label>
                <input type="number" id="capital_loss" name="capital_loss" class="form-input" value="{default_user_profile_template.get('capital_loss', 0)}" required>

                <label for="hours_per_week" class="form-label">Hours per Week:</label>
                <input type="number" id="hours_per_week" name="hours_per_week" class="form-input" value="{default_user_profile_template.get('hours_per_week', 40)}" required>

                <label for="native_country" class="form-label">Native Country:</label>
                <select id="native_country" name="native_country" class="form-select" required>
                    {''.join([f'<option value="{opt}" {"selected" if opt == default_user_profile_template.get("native_country") else ""}>{opt}</option>' for opt in categorical_feature_options.get('native_country', [])])}
                </select>

                <button type="submit" class="form-button w-full">Get Recommendations</button>
            </form>

            <div id="recommendationsOutput" class="recommendation-card hidden">
                <h2 class="text-xl font-semibold mb-4 text-gray-700">Recommendations:</h2>
                <div id="recommendationsList"></div>
            </div>
            <div id="errorMessage" class="text-red-600 mt-4 text-center hidden"></div>
        </div>

        <script>
            document.getElementById('recommendationForm').addEventListener('submit', async function(event) {{
                event.preventDefault(); // Prevent default form submission

                const form = event.target;
                const formData = new FormData(form);
                const userData = {{}};

                for (let [key, value] of formData.entries()) {{
                    // Convert numerical fields to numbers
                    if (['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'].includes(key)) {{
                        userData[key] = parseFloat(value);
                    }} else {{
                        userData[key] = value;
                    }}
                }}

                const recommendationsOutput = document.getElementById('recommendationsOutput');
                const recommendationsList = document.getElementById('recommendationsList');
                const errorMessage = document.getElementById('errorMessage');

                recommendationsList.innerHTML = ''; // Clear previous recommendations
                errorMessage.innerHTML = ''; // Clear previous error messages
                recommendationsOutput.classList.add('hidden');
                errorMessage.classList.add('hidden');

                try {{
                    const response = await fetch('/recommend', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify(userData)
                    }});

                    if (!response.ok) {{
                        const errorData = await response.json();
                        throw new Error(errorData.error || 'Something went wrong on the server.');
                    }}

                    const recommendations = await response.json();

                    if (recommendations.length > 0) {{
                        recommendations.forEach((rec, index) => {{
                            const item = document.createElement('p');
                            item.className = 'recommendation-item text-gray-700';
                            item.innerHTML = `<strong>${{index + 1}}. Change ${{rec.feature.replace('_', ' ')}}</strong> from '${{rec.original_value}}' to '<strong>${{rec.recommended_value}}</strong>'. <br>(Predicted probability of &gt;50K would increase by ${{rec.predicted_prob_increase.toFixed(2)}} to ${{rec.new_predicted_prob.toFixed(2)}})`;
                            recommendationsList.appendChild(item);
                        }});
                        recommendationsOutput.classList.remove('hidden');
                    }} else {{
                        recommendationsList.innerHTML = '<p class="text-gray-700">No significant recommendations found for this profile.</p>';
                        recommendationsOutput.classList.remove('hidden');
                    }}

                }} catch (error) {{
                    console.error('Error fetching recommendations:', error);
                    errorMessage.textContent = `Error: ${{error.message}}`;
                    errorMessage.classList.remove('hidden');
                }}
            }});
        </script>
    </body>
    </html>
    """
    return render_template_string(form_html)

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

    # Create a base profile by copying the default template
    # and then updating it with user-provided data
    current_profile = default_user_profile_template.copy()

    for key, value in user_data.items():
        if key in current_profile:
            current_profile[key] = value
        else:
            print(f"Warning: User provided unknown key '{key}'. Ignoring.")

    # Create a DataFrame from the combined profile, ensuring column order
    # This is crucial for the preprocessor
    user_profile_df = pd.DataFrame([current_profile], columns=all_feature_columns)

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


    ### to run the program please type $ /workspaces/Recommendation-Systems-Roza/.venv/bin/python /workspaces/Recommendation-Systems-Roza/src/app.py ###