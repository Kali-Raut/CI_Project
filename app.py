from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Preprocess input data
def preprocess_input(data):
    # Define mappings for categorical features
    mapping = {
        'Low': 0,
        'Medium': 1,
        'High': 2,
        'Yes': 1,
        'No': 0,
        'Negative': -1,
        'Neutral': 0,
        'Positive': 1,
        'Public': 0,
        'Private': 1,
        'Male': 0,
        'Female': 1,
        'High School': 0,
        'College': 1,
        'Postgraduate': 2,
        'Near': 0,
        'Moderate': 1,
        'Far': 2
    }

    # Apply mapping to categorical features
    data['Parental_Involvement'] = mapping[data['Parental_Involvement']]
    data['Access_to_Resources'] = mapping[data['Access_to_Resources']]
    data['Extracurricular_Activities'] = mapping[data['Extracurricular_Activities']]
    data['Motivation_Level'] = mapping[data['Motivation_Level']]
    data['Internet_Access'] = mapping[data['Internet_Access']]
    data['Family_Income'] = mapping[data['Family_Income']]
    data['Teacher_Quality'] = mapping[data['Teacher_Quality']]
    data['School_Type'] = mapping[data['School_Type']]
    data['Peer_Influence'] = mapping[data['Peer_Influence']]
    data['Learning_Disabilities'] = mapping[data['Learning_Disabilities']]
    data['Parental_Education_Level'] = mapping[data['Parental_Education_Level']]
    data['Distance_from_Home'] = mapping[data['Distance_from_Home']]
    data['Gender'] = mapping[data['Gender']]

    return data

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        input_data = {
            'Hours_Studied': int(request.form['Hours_Studied']),
            'Attendance': int(request.form['Attendance']),
            'Parental_Involvement': request.form['Parental_Involvement'],
            'Access_to_Resources': request.form['Access_to_Resources'],
            'Extracurricular_Activities': request.form['Extracurricular_Activities'],
            'Sleep_Hours': int(request.form['Sleep_Hours']),
            'Previous_Scores': int(request.form['Previous_Scores']),
            'Motivation_Level': request.form['Motivation_Level'],
            'Internet_Access': request.form['Internet_Access'],
            'Tutoring_Sessions': int(request.form['Tutoring_Sessions']),
            'Family_Income': request.form['Family_Income'],
            'Teacher_Quality': request.form['Teacher_Quality'],
            'School_Type': request.form['School_Type'],
            'Peer_Influence': request.form['Peer_Influence'],
            'Physical_Activity': int(request.form['Physical_Activity']),
            'Learning_Disabilities': request.form['Learning_Disabilities'],
            'Parental_Education_Level': request.form['Parental_Education_Level'],
            'Distance_from_Home': request.form['Distance_from_Home'],
            'Gender': request.form['Gender']
        }

        # Preprocess the input data
        processed_data = preprocess_input(input_data)

        # Convert processed data to a list of values
        input_features = list(processed_data.values())

        # Make a prediction using the model
        prediction = model.predict([input_features])[0]

        # Render the result on the page
        return render_template('index.html', prediction_text=f'Predicted Score: {prediction}')

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
