import os
import joblib
import numpy as np

# Make sure the current working directory is the package utilities directory so
# any relative model paths inside the utilities (e.g. "../models/position_classifier.pkl")
# resolve correctly. Adjust the path below if your repo is located elsewhere.
os.chdir('/Users/chanakyd/work/vdark/badminton/badminton/utilities')

# Load the trained artifacts
clf = joblib.load('../models/position_classifier.pkl')
scaler = joblib.load('../models/position_scaler.pkl')
le = joblib.load('../models/position_label_encoder.pkl')

def classify_triplet(point1, point2, point3):
    """
    Classify a set of three points into one of the base classes.

    Args:
        point1: Tuple of (x, y) for the first point (ignored since always origin).
        point2: Tuple of (x, y) for the second point.
        point3: Tuple of (x, y) for the third point.

    Returns:
        label: Predicted class label, e.g. "P_0_30"
    """
    # Extract coordinates as feature vector
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    
    #we want x1 and y1 to be 0,0 so sclae the 
    # features relative to the first point
    x2 -= x1
    y2 -= y1
    x3 -= x1
    y3 -= y1
        
    features = np.array([[x2, y2, x3, y3]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict and decode label
    pred_idx = clf.predict(features_scaled)[0]
    return le.inverse_transform([pred_idx])[0]

# Example usage
if __name__ == "__main__":
    # Define a sample triplet
    sample = ((0, 0), (43.3, 25.0), (68.3, 68.3))  # e.g., around P_30_90
    predicted_label = classify_triplet(*sample)
    print(f"Predicted label: {predicted_label}")
