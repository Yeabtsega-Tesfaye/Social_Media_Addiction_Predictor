from flask import Flask, request, render_template
from utils import predict_addiction

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Build feature dictionary from form inputs
        features = {
            'Age': int(request.form['age']),
            'Gender': request.form['gender'],
            'Academic_Level': request.form['academic_level'],
            'Country': request.form['country'],
            'Avg_Daily_Usage_Hours': float(request.form['usage_hours']),
            'Most_Used_Platform': request.form['platform'],
            'Affects_Academic_Performance': request.form['affects_academic'],
            'Sleep_Hours_Per_Night': float(request.form['sleep_hours']),
            'Mental_Health_Score': float(request.form['mental_health']),
            'Relationship_Status': request.form['relationship']
        }
        # Predict
        prediction = predict_addiction(features)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)