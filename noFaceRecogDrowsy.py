import cv2
import dlib
import io
import pygame
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from collections import deque
from imutils import face_utils
from threading import Thread, Event
import threading
import imutils
import skfuzzy as fuzz
import time
from playsound import playsound
from skfuzzy import control as ctrl
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from parser import get_args
from utils import get_landmarks, load_camera_parameters
from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from pydub import AudioSegment
from pydub.playback import play
from pydub.playback import play
import simpleaudio as sa

# Paths
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
alarm_path = "alarm.wav"
looking_path = "lookaway.mp3"
distracted_path = "focus.mp3"
yawn_path = "yawn.mp3"
drowsy_path = "drowsy2_alert.wav"


# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Initialize pygame mixer
pygame.mixer.init()

ALARM_ON = False
once = False
aggressive = False

warning = None
caliMessage = None
yawning = 0

lower_limit = 4
higher_limit = 8

# EAR and Blink Rate Constants
calibration_frames = 100
ear_values = deque(maxlen=calibration_frames)
blink_values = deque(maxlen=calibration_frames)
mar_values = deque(maxlen=calibration_frames)
calibrated_ear = False
calibrated_blink_rate = False
calibrated_mar = False
personal_eye_threshold = 0.25
personal_blink_rate_threshold = 15  # Blinks per minute default threshold
personal_mar_threshold = 20  # Default threshold for MAR

# Yawn Constants
yawn_threshold = 40  # Distance threshold for yawning
yawn_detected = False
yawns = 0

# Blink Variables

close_du=0
# start_time=time.time()
start_time = None
blink_rate_per_minute=0
COUNTER = 0
blink_count = 0
blink = 0
last_blink_time = time.time()
blink_start_time = time.time()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

args = get_args()

if args.camera_params:
    camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params)
else:
    camera_matrix, dist_coeffs = None, None

# Instantiate mediapipe face mesh model
Detector = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
)

# instantiation of the Eye Detector
Eye_det = EyeDet(show_processing=args.show_eye_proc)

# Instantiate the Head Pose estimator
Head_pose = HeadPoseEst(
    show_axis=args.show_axis, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
)

prev_time = time.perf_counter()

t_now = time.perf_counter()

# instantiation of the attention scorer object, with the various thresholds
# NOTE: set verbose to True for additional printed information about the scores
Scorer = AttScorer(
    t_now=t_now,
    gaze_time_thresh=args.gaze_time_thresh,
    roll_thresh=args.roll_thresh,
    pitch_thresh=args.pitch_thresh,
    yaw_thresh=args.yaw_thresh,
    gaze_thresh=args.gaze_thresh,
    pose_time_thresh=args.pose_time_thresh,
    verbose=args.verbose,
    )

def play_sound_continuous(file_path, state):
    def play_loop():
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        interval = max(0, 10.0 - state)  # Ensure interval is not negative
        try:
            while True:
                # Play the sound
                pygame.mixer.music.play()
                print("Playing sound...")
                
                # Wait until the sound finishes
                while pygame.mixer.music.get_busy():  # Check if sound is still playing
                    time.sleep(0.01)  # Frequent checks for short audio
                
                print("Sound finished, waiting for interval...")
                # Stop the sound (although it should already be stopped after it finishes)
                pygame.mixer.music.stop()
                
                # Wait for the interval
                time.sleep(interval)
                print(f"Interval of {interval} seconds finished. Replaying sound.")
        except KeyboardInterrupt:
            print("Playback stopped.")
            pygame.mixer.music.stop()
            pygame.mixer.quit()

    sound_thread = threading.Thread(target=play_loop)
    sound_thread.daemon = True  # Ensure the thread will close when the main program exits
    sound_thread.start()

def frequency_alarm(alarm_path, physical_state):
    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the sound
    sound = AudioSegment.from_file(alarm_path)
    
    # Adjust the playback speed based on the physical_state
    speed_adjustment = 1 + (physical_state - 1) * 0.1  # Adjust as needed
    sound = sound._spawn(sound.raw_data, overrides={"frame_rate": int(sound.frame_rate * speed_adjustment)})
    sound = sound.set_frame_rate(44100)  # Reset to standard frame rate if needed

    # Export the adjusted sound to a bytes buffer
    sound_buffer = io.BytesIO()
    sound.export(sound_buffer, format="wav")
    sound_buffer.seek(0)

    # Load the sound into pygame mixer
    sound_effect = pygame.mixer.Sound(sound_buffer)

    # Play the sound
    sound_effect.play()

    # Keep the sound playing in the background
    while pygame.mixer.get_busy():
        pygame.time.delay(100)

    # Quit the mixer after playing
    pygame.mixer.quit()


def frequency_alarm_4(alarm_path, physical_state):
    """
    Play an alarm sound with dynamically adjusted pauses based on physical state.

    Parameters:
    - alarm_path: Path to the alarm sound file.
    - physical_state_func: Function to retrieve the current physical state.
    """
    # Load the sound
    sound_effect = pygame.mixer.Sound(alarm_path)

    while ALARM_ON:

        # Calculate the delay (10 - physical_state)
        delay = 10.0 - physical_state
        delay = max(0.0, delay)  # Ensure a minimum delay of 0.1 seconds

        # Play the sound once
        sound_effect.play()

        # Allow the sound to play completely before applying the delay
        while pygame.mixer.get_busy():
            time.sleep(0.02)

        # Wait for the calculated delay before playing the sound again
        time.sleep(delay)

def frequency_alarm_3(alarm_path, physical_state):
    """
    Play an alarm sound with dynamically adjusted pauses based on physical state.

    Parameters:
    - alarm_path: Path to the alarm sound file.
    - physical_state_func: Function to retrieve the current physical state.
    """
    # Load the sound
    sound_effect = pygame.mixer.Sound(alarm_path)

    while ALARM_ON:
        # # Retrieve the current physical state
        # physical_state = physical_state_func()

        # Calculate the delay (10 - physical_state)
        delay = 10 - physical_state
        delay = max(0.1, delay)  # Ensure a minimum delay of 0.1 seconds

        # Play the sound
        sound_effect.play()

        # Allow the sound to play completely (assuming it's short)
        while pygame.mixer.get_busy():
            time.sleep(0.1)

        # Wait for the calculated delay before playing the sound again
        time.sleep(delay)

def frequency_alarm_5(sound_path, physical_state):
    """
    Play a sound continuously with a specified time interval between replays.

    Parameters:
    - sound_path: Path to the sound file.
    - interval: Time interval (in seconds) between replays. Default is 10 seconds.
    """
    # Load the sound
    sound_effect = pygame.mixer.Sound(sound_path)
    interval = 10-physical_state

    while True:
        # Play the sound
        sound_effect.play()

        # Wait for the sound to finish playing
        while pygame.mixer.get_busy():
            time.sleep(0.1)

        # Wait for the specified interval before replaying
        time.sleep(interval)

def play_sound_with_dynamic_interval(sound_path, state):
    """
    Play a short sound with a dynamically adjusted interval between plays based on physical state.

    Parameters:
    - sound_path: Path to the short sound file.
    - get_physical_state_func: A function to retrieve the current physical state.
    """
    # Load the sound
    sound_effect = pygame.mixer.Sound(sound_path)

    while True:

        # Calculate the interval (10 - physical_state)
        interval = max(0.1, 10 - state)  # Ensure a minimum interval of 0.1 seconds

        # Play the sound
        sound_effect.play()

        # Wait for the sound to finish playing
        while pygame.mixer.get_busy():
            time.sleep(0.01)

        # Wait for the calculated interval
        time.sleep(interval)

def intensity_alarm(alarm_path, intensity):
    # Initialize pygame mixer
    pygame.mixer.init()

    # Load and adjust the sound
    sound = AudioSegment.from_file(alarm_path)
    volume_adjustment = (intensity - 10) * 3  # Adjusting the scale
    sound = sound + volume_adjustment

    # Export the adjusted sound to a bytes buffer
    sound_buffer = io.BytesIO()
    sound.export(sound_buffer, format="wav")
    sound_buffer.seek(0)

    # Load the sound into pygame mixer
    sound_effect = pygame.mixer.Sound(sound_buffer)

    # Play the sound
    sound_effect.play()

    # Keep the sound playing in the background
    while pygame.mixer.get_busy():
        pygame.time.delay(100)

    # Quit the mixer after playing
    # pygame.mixer.quit()

def sound_alarm(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    # # Quit the mixer after playing
    # pygame.mixer.quit()

def fuzzyRules():
    # Define fuzzy variables for EAR (eye state) and MAR (mouth state)
    eyeblinkf = ctrl.Antecedent(np.arange(0, 25, 1), 'blink_count')
    closured = ctrl.Antecedent(np.arange(0, 30, 1), 'close_du')
    driverstate=ctrl.Consequent(np.arange(0, 10, 1), 'driverstate')

    # Membership functions for eye blink frequency
    eyeblinkf['short'] = fuzz.trimf(eyeblinkf.universe, [0, 0, 4])
    eyeblinkf['medium'] = fuzz.trimf(eyeblinkf.universe, [3, 8, 13])
    eyeblinkf['long'] = fuzz.trimf(eyeblinkf.universe, [8, 25, 25])

    # Membership functions for eye closure duration
    closured['short'] = fuzz.trimf(closured.universe, [0, 0, 5])
    closured['medium'] = fuzz.trimf(closured.universe, [4, 10, 20])
    closured['long'] = fuzz.trimf(closured.universe, [8, 30, 30])

    # Membership functions for driver physical state
    driverstate['safe'] = fuzz.trimf(driverstate.universe, [0, 0, 4])
    driverstate['caution'] = fuzz.trimf(driverstate.universe, [3, 5, 9])
    driverstate['danger'] = fuzz.trimf(driverstate.universe, [8, 10, 10])

    rule1 = ctrl.Rule(eyeblinkf['short'] & closured['short'], driverstate['safe'])
    rule2 = ctrl.Rule(eyeblinkf['short'] & closured['medium'], driverstate['safe'])
    rule3 = ctrl.Rule(eyeblinkf['short'] & closured['long'], driverstate['caution'])
    rule4 = ctrl.Rule(eyeblinkf['medium'] & closured['short'], driverstate['caution'])
    rule5 = ctrl.Rule(eyeblinkf['medium'] & closured['medium'], driverstate['caution'])
    rule6 = ctrl.Rule(eyeblinkf['medium'] & closured['long'], driverstate['danger'])
    rule7 = ctrl.Rule(eyeblinkf['long'] & closured['short'], driverstate['danger'])
    rule8 = ctrl.Rule(eyeblinkf['long'] & closured['medium'], driverstate['danger'])
    rule9 = ctrl.Rule(eyeblinkf['long'] & closured['long'], driverstate['danger'])

    # Create control system
    control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    fuzzy_evaluator = ctrl.ControlSystemSimulation(control_system)

    return fuzzy_evaluator

# EAR Calculation Function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Lip distance calculation for yawns
def top_lip(landmarks):
    return int(np.mean([landmarks[i][1] for i in range(50, 53)] + [landmarks[i][1] for i in range(61, 64)]))

def bottom_lip(landmarks):
    return int(np.mean([landmarks[i][1] for i in range(65, 68)] + [landmarks[i][1] for i in range(56, 59)]))

def calculate_lip_distance(landmarks):
    return abs(top_lip(landmarks) - bottom_lip(landmarks))

# Calibration Functions
def calibrate_ear(ear_values):
    if len(ear_values) > 0:
        mean_ear = np.mean(ear_values)
        return mean_ear * 0.9  # Set threshold as 85% of mean EAR
    return 0.25

def calibrate_blink_rate(blink_values):
    if len(blink_values) > 0:
        mean_blink_rate = np.mean(blink_values)
        return mean_blink_rate * 1.2  # Set threshold as 120% of mean blink rate
    return 15

def calibrate_mar(mar_values):
    if len(mar_values) > 0:
        mean_mar = np.mean(mar_values)
        return mean_mar * 1.1  # Set threshold as 110% of mean MAR
    return 20

def earmarCalc(shape_np, frame):
    # EAR Calculation
    leftEye = shape_np[lStart:lEnd]
    rightEye = shape_np[rStart:rEnd]
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
    
    lip_distance = calculate_lip_distance(shape_np)

    return ear, lip_distance
    

def overallCali(frame, ear, lip_distance):
    global calibrated_ear, calibrated_blink_rate, calibrated_mar, COUNTER, blink_count
    global personal_eye_threshold, personal_blink_rate_threshold, personal_mar_threshold, caliMessage

    # First Calibration: EAR Threshold Calculation
    if not calibrated_ear:
        #print('[INFO] Start Calibrating EAR...')
        ear_values.append(ear)
        cv2.putText(frame, "Calibrating EAR...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        caliMessage = "Take your few time for Calibrating..."

        if len(ear_values) == calibration_frames:
            personal_eye_threshold = calibrate_ear(ear_values)
            calibrated_ear = True
            print(f"[INFO] EAR Calibration complete. EAR threshold: {personal_eye_threshold:.2f}")

    # Second Calibration: Blink Rate Threshold Calculation
    # Blink Rate Calibration
    elif calibrated_ear and not calibrated_blink_rate:
        #print('[INFO] Start Calibrating Blink Rate...')
        current_time = time.time()

        # Detect blinks based on EAR threshold
        if ear < personal_eye_threshold:
            COUNTER += 1
        else:
            if COUNTER >= 2:  # A valid blink is detected
                blink_count += 1
                COUNTER = 0  # Reset counter
                last_blink_time = time.time()

        # Append the blink count to the list if within calibration phase
        if len(blink_values) < calibration_frames:
            blink_values.append(blink_count)
            cv2.putText(frame, "Calibrating blink rate...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            caliMessage = "Take your few time for Calibrating..."
        else:
            # Stop calibration after enough frames
            calibrated_blink_rate = True
            personal_blink_rate_threshold = calibrate_blink_rate(blink_values)
            print(f"[INFO] Blink Rate Calibration complete. Blink Rate threshold: {personal_blink_rate_threshold:.2f}")

    # Calibration for MAR
    elif calibrated_ear and calibrated_blink_rate and not calibrated_mar:
        #print('[INFO] Start Calibrating MAR...')
        mar_values.append(lip_distance)
        cv2.putText(frame, "Calibrating MAR...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        caliMessage = "Take your few time for Calibrating..."
        if len(mar_values) == calibration_frames:
            personal_mar_threshold = calibrate_mar(mar_values)
            calibrated_mar = True
            print(f"[INFO] MAR Calibration complete. MAR threshold: {personal_mar_threshold:.2f}")

    return personal_eye_threshold, personal_blink_rate_threshold, personal_mar_threshold, caliMessage

def drowsyDetection(ear, lip_distance, personal_eye_threshold, personal_blink_rate_threshold, personal_mar_threshold):
    global COUNTER, start_time, closure_duration, blink_start_time, close_du, yawns,blink_rate_per_minute, yawn_detected,blink
    global warning, yawning

    #blink = 0
    if ear < personal_eye_threshold:
        COUNTER += 1
        if start_time is None:
            start_time = time.time()  # Start timing
    else:
        if COUNTER >= 2:
            blink += 1
            if start_time is not None:
                # Add duration of eye closure
                closure_duration = time.time() - start_time
                close_du = closure_duration
                start_time = None  # Reset timer
        COUNTER = 0

    # Yawn Detection Logic
    if lip_distance > (3.5*personal_mar_threshold):
        if not yawn_detected:
            yawns += 1
            yawning += 1
            yawn_detected = True
    else:
        yawn_detected = False


    if yawning > 5:
        ALARM_ON = False
        if not ALARM_ON:
            warning = "Frequent yawns detected. Please take a break!"
            ALARM_ON = True
            
            if yawn_path:
                t = Thread(target=sound_alarm, args=(yawn_path,))
                t.daemon = True
                t.start()
        #once = True
        yawning = 0
    else:
        ALARM_ON=False
        # once = False
        warning = None

    # Calculate Blink Rate per Minute
    current_time = time.time()
    if current_time - blink_start_time >= 60:  # 1 minute passed
        blink_rate_per_minute = blink_count / 60  # Total blinks in 1 minute
        blink_start_time = current_time  # Reset the timer
        blink = 0  # Reset blink count
        yawns = 0
        yawning = 0

    # Fuzzy Logic Evaluation
    avg_blink_rate = np.mean(blink_values) if len(blink_values) > 0 else personal_blink_rate_threshold

    return blink, close_du, yawns, blink_rate_per_minute, warning

def headPoseEstimation(frame,landmarks,gray,frame_size,t_now):
    global ALARM_ON
    # shows the eye keypoints (can be commented)
    Eye_det.show_eye_keypoints(
        color_frame=frame, landmarks=landmarks, frame_size=frame_size
    )

    # compute the Gaze Score
    gaze = Eye_det.get_Gaze_Score(
        frame=gray, landmarks=landmarks, frame_size=frame_size
    )

    # compute the head pose
    frame_det, roll, pitch, yaw = Head_pose.get_pose(
        frame=frame, landmarks=landmarks, frame_size=frame_size
    )

    # evaluate the scores for EAR, GAZE and HEAD POSE
    looking_away, distracted = Scorer.eval_scores(
        t_now=t_now,
        gaze_score=gaze,
        head_roll=roll,
        head_pitch=pitch,
        head_yaw=yaw,
    )

    # if the head pose estimation is successful, show the results
    if frame_det is not None:
        frame = frame_det

    # show the real-time Gaze Score
    if gaze is not None:
        cv2.putText(
            frame,
            "gaze score:" + str(round(gaze, 3)),
            (400, 30),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    if roll is not None:
        cv2.putText(
            frame,
            "roll:" + str(roll.round(1)[0]),
            (400, 50),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    if pitch is not None:
        cv2.putText(
            frame,
            "pitch:" + str(pitch.round(1)[0]),
            (400, 70),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    if yaw is not None:
        cv2.putText(
            frame,
            "yaw:" + str(yaw.round(1)[0]),
            (400, 90),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    warning1 = False
    warning2 = False
    # warning = "Safe Driving"
    # if the state of attention of the driver is not normal, show an alert on screen
    if looking_away:
        warning1 = True
        cv2.putText(
            frame,
            "LOOKING AWAY!",
            (10, 320),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        if not ALARM_ON:
            ALARM_ON = True
            if looking_path:
                t = Thread(target=sound_alarm, args=(looking_path,))
                t.deamon = True
                t.start() 
    else:
        ALARM_ON=False
        #warning = "Safe Driving"
        warning1 = False


    if distracted:
        # if warning:
        #     warning += " | Please stay focused!"
        # else:
        #     warning = "Please stay focused!"
        #warning2 = "Please stay focused!"
        warning2= True
        ALARM_ON = False
        cv2.putText(
            frame,
            "DISTRACTED!",
            (10, 340),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        if not ALARM_ON:
            ALARM_ON = True
            
            if distracted_path:
                t = Thread(target=sound_alarm, args=(distracted_path,))
                t.deamon = True
                t.start()
    else:
        ALARM_ON=False
        #warning = "Safe Driving"
        warning2 = False

    if warning1:
        warning = "Please keep your eyes on the road!" # Assign warning from drowsy detection
    elif warning2:
        warning = "Please stay focused!"  # Assign warning from head pose estimation
    else:
        warning = "Safe Driving"
     
    return warning

def play_sound_continuously(file_path, state):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    interval = 10.0 - state
    try:
        while True:
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)  # Check more frequently if the sound has finished playing
            time.sleep(interval)  # Wait for the specified interval (can be decimal)
    except KeyboardInterrupt:
        print("Playback stopped.")
        pygame.mixer.music.stop()

def play_sound_repeatedly(sound_file, state):
    pygame.mixer.init()  # Initialize the mixer module
    sound = pygame.mixer.Sound(sound_file)  # Load the sound file

    def play():
        for _ in range(5):
            sound.play()  # Play the sound
            while pygame.mixer.get_busy():  # Wait for the sound to finish playing
                time.sleep(0.1)  # Check periodically

            interval = max(0.1, 10 - state)  # Get the dynamic interval
            print(f"Waiting for {interval} seconds before playing again...")
            time.sleep(interval)  # Pause for the interval duration

        pygame.mixer.quit()  # Quit the mixer when done

    # Start the sound playing in a new thread
    sound_thread = threading.Thread(target=play)
    sound_thread.start()
      
   


def main():
    global ALARM_ON, warning, messagecali, alarm_thread_event

    fuzzy_evaluator = fuzzyRules()
    driver_missing_counter = 0
    physical_state = None 
    landmarks = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()
    print("[INFO] Starting video stream...")

    while True:
        # get current time in seconds
        t_now = time.perf_counter()

        ret, frame = cap.read()  # read a frame from the webcam
        if not ret: # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

        frame = imutils.resize(frame, width=600)

        # if the frame comes from webcam, flip it so it looks like a mirror.
        # if args.camera == 0:
        frame = cv2.flip(frame, 2)

        #convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = np.expand_dims(gray, axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)
        
        # get the frame size (width and height)
        frame_size = frame.shape[1], frame.shape[0]

        # get the frame size
        frame_size = frame.shape[1], frame.shape[0]

        # Find faces using the face mesh model
        lms = Detector.process(gray).multi_face_landmarks

        if lms:  # Process the frame only if at least a face is found
            # Get landmarks
            landmarks = get_landmarks(lms)
            
        #detect face using dlib's frontal face detector
        rects = detector(gray)
            #loop for each detected face
        if rects:
            driver_missing_counter = 0 
            # Find the largest rectangle by area
            largest_face = max(rects, key=lambda rect: rect.width() * rect.height())

            # Process only the largest face
            x, y, w, h = largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height()
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            shape = predictor(gray, largest_face)
            shape_np = face_utils.shape_to_np(shape) #landmarks
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Assuming `shape` is the object returned by dlib's facial landmark detector
            for part in shape.parts():
                x, y = part.x, part.y
                # print(f"Landmark point: ({x}, {y})")
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            #assign returned tuple
            earmarCalc(shape_np, frame)
            ear, lip_distance = earmarCalc(shape_np, frame)
            
            overallCali(frame,ear,lip_distance)
            personal_eye_threshold, personal_blink_rate_threshold, personal_mar_threshold, messagecali = overallCali(frame, ear, lip_distance)

            if messagecali:
                    warning = messagecali

            if calibrated_ear and calibrated_blink_rate and calibrated_mar:
                # drowsyDetection(ear, lip_distance, personal_eye_threshold, personal_blink_rate_threshold, personal_mar_threshold)
                # headPoseEstimation(frame, landmarks, gray, frame_size, t_now)
                # blink_count, close_du, yawns, blink_rate_per_minute, warning = drowsyDetection(ear, lip_distance,personal_eye_threshold, personal_blink_rate_threshold, personal_mar_threshold)
                # warning = headPoseEstimation(frame, landmarks, gray, frame_size, t_now)

                blink_count, close_du, yawns, blink_rate_per_minute, drowsy_warning = drowsyDetection(ear, lip_distance, personal_eye_threshold, personal_blink_rate_threshold, personal_mar_threshold)
                headPoseEstimation(frame, landmarks, gray, frame_size, t_now)
                # Capture both warnings
                warning_from_pose = headPoseEstimation(frame, landmarks, gray, frame_size, t_now)

                # Combine the warnings or handle them separately
                if drowsy_warning:
                    warning = drowsy_warning  # Assign warning from drowsy detection
                elif warning_from_pose:
                    warning = warning_from_pose  # Assign warning from head pose estimation
                else:
                    warning = "Safe Driving"

                # Apply fuzzy logic

                fuzzy_evaluator.input['blink_count'] = blink_count
                fuzzy_evaluator.input['close_du'] = close_du
                fuzzy_evaluator.compute()

                physical_state = fuzzy_evaluator.output['driverstate']

                # audio = AudioSegment.from_file(drowsy_path)

                # # Change tempo by altering the frame rate
                # def change_tempo(audio, speed=1.0):
                #     new_frame_rate = int(audio.frame_rate * speed)
                #     return audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate}).set_frame_rate(audio.frame_rate)
                
                # # Increase tempo by 1.5x
                # faster_audio = change_tempo(audio, (max(0.0,(physical_state/2))))

                # if(physical_state>2):

                #     # Play the modified audio
                #     play(faster_audio)

               
                if physical_state > 2:
                    if not ALARM_ON:
                        ALARM_ON = True
                        if alarm_path:
                            t = Thread(target=intensity_alarm, args=(drowsy_path, physical_state))
                            t.daemon = True
                            t.start()
                else:
                    ALARM_ON = False

                def get_driver_state_label(driver_state_value):
                    if driver_state_value <= 4:
                        return "Low"
                    elif 4 < driver_state_value <= 8:
                        return "Medium"
                    else:
                        return "High"
                    
                # Get linguistic term for driver state
                driver_state_label = get_driver_state_label(physical_state)

                # Display the linguistic term
                cv2.putText(frame, f"Drowsiness state: {driver_state_label}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display Results
                cv2.putText(frame, f"EAR: {ear:.2f}", (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, f"MAR: {lip_distance:.2f}", (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, f"Blink Rate: {blink_rate_per_minute:.2f} BPM", (180, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, f"Close for: {close_du:.2f}", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                cv2.putText(frame, f"Blink Count: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, f"Yawns: {yawns}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        else:
            driver_missing_counter += 1
            if driver_missing_counter>5:
                warning = "Driver is MISSING!"
                if not ALARM_ON:
                    ALARM_ON = True
                    if alarm_path:
                        t = Thread(target=sound_alarm, args=(alarm_path,))
                        t.deamon = True
                        t.start()
                cv2.putText(
                frame,
                "DRIVER MISSING!",
                (10, 300),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA
                )
            else:
                ALARM_ON=False
                warning = "Safe Driving"

        print(warning)
        print(physical_state)
                
        # show the frame on screen
        cv2.imshow("Press 'q' to terminate", frame)

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    


    cap.release()
    cv2.destroyAllWindows()

    return {"warning": warning, "physical_state": physical_state}
    

if __name__ == "__main__":
    main()


#everything works very well