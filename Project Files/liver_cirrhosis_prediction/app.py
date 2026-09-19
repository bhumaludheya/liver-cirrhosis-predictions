from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and scaler
try:
    model = pickle.load(open("rf_acc_68.pkl", "rb"))
    scaler = pickle.load(open("normalizer.pkl", "rb"))
except Exception as e:
    print("❌ Error loading model or scaler:", e)
    exit()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert gender
        gender = 1 if request.form['gender'].lower() == 'male' else 0

        # Extract and order features
        features = [
            float(request.form['feature10']),  # Age
            gender,
            float(request.form['feature2']),   # Total Bilirubin
            float(request.form['feature3']),   # Direct Bilirubin
            float(request.form['feature4']),   # Alkaline Phosphotase
            float(request.form['feature5']),   # ALT
            float(request.form['feature6']),   # AST
            float(request.form['feature7']),   # Total Proteins
            float(request.form['feature8']),   # Albumin
            float(request.form['feature9'])    # A:G Ratio
        ]

        print("🧪 RAW INPUT:", features)

        scaled = scaler.transform([features])
        prediction = model.predict(scaled)
        print("🎯 MODEL PREDICTION:", prediction)

        result = "Cirrhosis Detected" if prediction[0] == 1 else "No Cirrhosis Detected"
        return render_template("portfolio-details.html", prediction=result)

    except Exception as e:
        print("❌ Error during prediction:", e)
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
