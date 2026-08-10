import sys
import os
sys.path.append(os.path.dirname(__file__))
import sqlite3
import cv2
from fastapi import FastAPI, UploadFile, File
from model_builder import build_model
import numpy as np
import keras

app = FastAPI()

# 1. Initialize Database
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scans 
                 (id INTEGER PRIMARY KEY, result TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_scan_to_db(result):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("INSERT INTO scans (result) VALUES (?)", (result,))
    conn.commit()
    conn.close()

# Run database initialization
init_db()

# 2. Load Model
model = build_model(input_shape=(224, 224, 3), num_classes=3)
model.load_weights('best_xception_model.keras')

# 3. Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("--- Request received! ---")
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "Error", "message": "Could not decode image"}
            
        # 1. VALIDATION
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (img.shape[0] * img.shape[1])
        
        if edge_density > 10.0: 
            print(f"Rejected: High edge density ({edge_density:.2f}) - Likely not a Chest X-ray.")
            return {"status": "Success", "result": "Rejected: Invalid/Non-X-ray Image"}

        # 2. PREPROCESSING
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)
        
        # 3. MODEL PREDICTION
        prediction = model.predict(img_input)
        confidence_scores = prediction[0] 
        
        normal_score = float(confidence_scores[0])
        pneumonia_score = float(confidence_scores[1])
        
        # 4. FINAL DECISION LOGIC
        if pneumonia_score > normal_score:
            result = "Pneumonia"
        else:
            result = "Normal"
            
        # 5. SAVE TO DATABASE & RETURN
        save_scan_to_db(result)
        return {"status": "Success", "result": result}

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"status": "Error", "message": str(e)}

@app.get("/")
def home():
    return {"message": "Model is running correctly!"}
