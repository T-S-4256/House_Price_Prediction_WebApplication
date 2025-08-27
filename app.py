from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved model
model = pickle.load(open('models/house_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        MedInc = float(request.form['MedInc'])
        HouseAge = float(request.form['HouseAge'])
        AveRooms = float(request.form['AveRooms'])
        AveBedrms = float(request.form['AveBedrms'])
        Population = float(request.form['Population'])
        AveOccup = float(request.form['AveOccup'])
        Latitude = float(request.form['Latitude'])
        Longitude = float(request.form['Longitude'])
        
        # Create input array for model
        input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        return render_template('index.html', prediction_text=f'Estimated House Price: ${prediction}')
    return render_template('index.html')
@app.route('/healthz')
def healthz():
    return jsonify({"status": "ok"}), 200

# Serve robots.txt to avoid 404s in logs
@app.route('/robots.txt')
def robots_txt():
    return send_from_directory(app.static_folder, 'robots.txt')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render’s $PORT
    app.run(host='0.0.0.0', port=port, debug=True)

