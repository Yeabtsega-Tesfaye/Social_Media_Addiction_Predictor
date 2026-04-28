from flask import Flask, request, render_template, jsonify
from utils import predict_addiction

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receives JSON from frontend
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        result = predict_addiction(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)