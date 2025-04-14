from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import sys
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS

# Paths for models and dataset
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modified_datasets", "events.csv")

# Load trained models and preprocessors
print("Loading models from:", path)

kmeans: KMeans = joblib.load(os.path.join(path, "kmeans_model.joblib"))
dbscan: DBSCAN = joblib.load(os.path.join(path, "dbscan_model.joblib"))
gmm: GaussianMixture = joblib.load(os.path.join(path, "gmm_model.joblib"))
scaler: StandardScaler = joblib.load(os.path.join(path, "scaler.joblib"))
label_enc: LabelEncoder = joblib.load(os.path.join(path, "label_encoder.joblib"))

print("Models and preprocessors loaded successfully!")

# Load dataset
df = pd.read_csv(dataset_path)
print("Dataset loaded successfully:", df.shape)

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)

        # Convert JSON to DataFrame
        data_df = pd.DataFrame([data])

        # Extract input features
        price = float(data_df.loc[0, 'price'])
        day_of_week = int(data_df.loc[0, 'day_of_week'])
        event_type = data_df.loc[0, 'type']

        # Check if event type is known; otherwise, update LabelEncoder
        if event_type not in label_enc.classes_:
            new_classes = np.append(label_enc.classes_, event_type)
            label_enc.classes_ = new_classes
            print(f"Updated LabelEncoder classes: {label_enc.classes_}")

        # Encode event type correctly as a 1D array
        type_encoded = label_enc.transform([event_type])[0]

        # Scale numerical features
        features = pd.DataFrame([[price, day_of_week]], columns=['price', 'day_of_week'])
        features_scaled = scaler.transform(features)

        # Convert back to DataFrame
        features_scaled_df = pd.DataFrame(features_scaled, columns=['price', 'day_of_week'])
        features_scaled_df["type_encoded"] = type_encoded  # Append encoded event type

        # Get predictions
        kmeans_cluster = int(kmeans.predict(features_scaled_df)[0])
        gmm_cluster = int(gmm.predict(features_scaled_df)[0])
        dbscan_cluster = int(dbscan.fit_predict(features_scaled_df)[0])

        # Get recommended events (ignore DBSCAN cluster if it's -1)
        recommended_events = df[
            (df["KMeans_Cluster"] == kmeans_cluster) |
            (df["GMM_Cluster"] == gmm_cluster) |
            ((df["DBSCAN_Cluster"] == dbscan_cluster) & (dbscan_cluster != -1))
        ][["title", "type", "city"]].to_dict(orient="records")

        return jsonify({
            'KMeans_Cluster': kmeans_cluster,
            'DBSCAN_Cluster': dbscan_cluster,
            'GMM_Cluster': gmm_cluster,
            'recommended_events': recommended_events
        })

    except Exception as e:
        print(f"Flask Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Explicitly set port to 5001
