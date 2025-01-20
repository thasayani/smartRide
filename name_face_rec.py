import face_recognition
import cv2
import numpy as np
import json
import os

# Database file to store face encodings and names
database_file = "face_database.json"

# # Load database
# if os.path.exists(database_file):
#     try:
#         with open(database_file, "r") as db_file:
#             data = json.load(db_file)
#             known_face_encodings = [np.array(encoding) for encoding in data.get('encodings', [])]
#             known_face_names = data.get('names', [])
#     except (json.JSONDecodeError, ValueError):
#         known_face_encodings = []
#         known_face_names = []
# else:
#     known_face_encodings = []
#     known_face_names = []

def load_database():
    if os.path.exists(database_file):
        try:
            with open(database_file, "r") as db_file:
                data = json.load(db_file)
                known_face_encodings = [np.array(encoding) for encoding in data.get('encodings', [])]
                known_face_names = data.get('names', [])
                calibration_data = data.get('calibration_data', {})
                return known_face_encodings, known_face_names, calibration_data
        except (json.JSONDecodeError, ValueError):
            return [], [], {}
    else:
        return [], [], {}

def save_to_database(known_face_encodings, known_face_names, calibration_data):
    with open(database_file, "w") as db_file:
        json.dump({
            'encodings': [enc.tolist() for enc in known_face_encodings],
            'names': known_face_names,
            'calibration_data': calibration_data
        }, db_file)

# Get a reference to the webcam
# def recognize_face():
#     known_face_encodings, known_face_names, calibration_data = load_database()
#     video_capture = cv2.VideoCapture(0)

#     if not video_capture.isOpened():
#         print("Error: Could not access the camera.")
#         exit()

#     frame_counter = 0  # Counter for skipping frames

#     while True:
#         # Grab a frame from the video
#         ret, frame = video_capture.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break

#         # Process every nth frame
#         frame_counter += 1
#         if frame_counter % 5 != 0:  # Skip 4 out of 5 frames
#             continue

#         # Convert frame to RGB for face_recognition
#         rgb_frame = frame[:, :, ::-1]

#         # Detect faces and compute encodings
#         face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Faster model
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             # Compute distances to all known faces
#             distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
#             # Find the best match
#             best_match_index = np.argmin(distances) if len(distances) > 0 else -1
#             name = "Unknown"

#             if best_match_index != -1 and distances[best_match_index] < 0.6:  # Use a stricter threshold
#                 name = known_face_names[best_match_index]
#                 # print(f"Welcome back, {name}!")
#             else:
#                 # Prompt user to name the unknown face
#                 print("Unknown face detected! Please enter your name.")
#                 name = input("Enter your name: ")

#                 # Save new encoding and name
#                 known_face_encodings.append(face_encoding)
#                 known_face_names.append(name)

#         #         with open(database_file, "w") as db_file:
#         #             json.dump({'encodings': [enc.tolist() for enc in known_face_encodings], 'names': known_face_names}, db_file)
#         #         print(f"Face added for {name}")

#         #     # Draw a box around the face
#         #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         #     # Label the face
#         #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         #     font = cv2.FONT_HERSHEY_DUPLEX
#         #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#         # # Display the frame
#         # cv2.imshow('Video', frame)

#         # Exit on 'q'
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break

#             recognized_name = name
#             recognized_encoding = face_encoding

#             # Break out of the loop once the face is recognized and details are obtained
#             break

#         if recognized_name and recognized_encoding:
#             break

#     # Release the webcam and close windows
#     video_capture.release()
#     cv2.destroyAllWindows()

#     return recognized_name, recognized_encoding

def recognize_face():
    known_face_encodings, known_face_names, calibration_data = load_database()
    video_capture = cv2.VideoCapture(1)

    if not video_capture.isOpened():
        print("Error: Could not access the camera.")
        exit()

    frame_counter = 0  # Counter for skipping frames

    recognized_name = None
    recognized_encoding = None

    while True:
        # Grab a frame from the video
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Process every nth frame
        frame_counter += 1
        if frame_counter % 5 != 0:  # Skip 4 out of 5 frames
            continue

        # Convert frame to RGB for face_recognition
        rgb_frame = frame[:, :, ::-1]

        # Detect faces and compute encodings
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Faster model
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compute distances to all known faces
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
            # Find the best match
            best_match_index = np.argmin(distances) if len(distances) > 0 else -1
            name = "Unknown"

            if best_match_index != -1 and distances[best_match_index] < 0.6:  # Use a stricter threshold
                name = known_face_names[best_match_index]
                print(f"Welcome back, {name}!")
            else:
                # Prompt user to name the unknown face
                print("Unknown face detected! Please enter your name.")
                name = input("Enter your name: ")

                # Save new encoding and name
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

                with open(database_file, "w") as db_file:
                    json.dump({'encodings': [enc.tolist() for enc in known_face_encodings], 'names': known_face_names}, db_file)
                print(f"Face added for {name}")

            recognized_name = name
            recognized_encoding = face_encoding

            # Break out of the loop once the face is recognized and details are obtained
            break

        if recognized_name and recognized_encoding is not None and recognized_encoding.any():
            print("Start Monitoring")
            break

    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

    return recognized_name, recognized_encoding


#working good, just a bit lag