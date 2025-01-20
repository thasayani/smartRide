from flask import Flask, request, jsonify
import face_recognition as facedriver_recog
import numpy as np
import json
import base64
import cv2
import os

app = Flask(__name__)

# Database file to store face encodings and names
database_file = "face_database.json"

# Load database
if os.path.exists(database_file):
    try:
        with open(database_file, "r") as db_file:
            data = json.load(db_file)
            known_face_encodings = [np.array(encoding) for encoding in data.get('encodings', [])]
            known_face_names = data.get('names', [])
    except (json.JSONDecodeError, ValueError):
        known_face_encodings = []
        known_face_names = []
else:
    known_face_encodings = []
    known_face_names = []

@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        # Get the image from the request
        image_data = request.json['image']
        image_data = base64.b64decode(image_data)
        np_image = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        
        # Convert frame to RGB
        rgb_frame = frame[:, :, ::-1]
        
        # Detect faces and compute encodings
        face_locations = facedriver_recog.face_locations(rgb_frame, model="hog")
        face_encodings = facedriver_recog.face_encodings(rgb_frame, face_locations)
        
        response = {"message": "No face detected"}

        for face_encoding in face_encodings:
            # Compute distances to all known faces
            distances = facedriver_recog.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(distances) if len(distances) > 0 else -1
            name = "Unknown"

            if best_match_index != -1 and distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]
                response = {"message": f"Welcome back, {name}!"}
            else:
                response = {"message": "Hi, new user! Please enter your name."}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_user', methods=['POST'])
def add_user():
    try:
        name = request.json['name']
        encoding = np.array(request.json['encoding'])

        known_face_encodings.append(encoding)
        known_face_names.append(name)

        # Save updated database
        with open(database_file, "w") as db_file:
            json.dump({'encodings': [enc.tolist() for enc in known_face_encodings], 'names': known_face_names}, db_file)

        return jsonify({"message": f"Face added for {name}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
