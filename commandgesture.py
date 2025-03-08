import os
import cv2
import json
import base64
import webbrowser
import streamlit as st
import mediapipe as mp
import google.generativeai as genai
import urllib.parse
import subprocess

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDZrMVYfS1AhFagS6cafqAu1NTqnmoFggc")
if not API_KEY:
    st.error("API Key is missing! Set GEMINI_API_KEY as an environment variable.")
    st.stop()

genai.configure(api_key=API_KEY)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load gestures from JSON file
GESTURE_FILE = "gestures.json"
if os.path.exists(GESTURE_FILE):
    with open(GESTURE_FILE, "r") as file:
        gesture_mappings = json.load(file)
else:
    gesture_mappings = {}

# Save gestures to JSON
def save_gestures():
    """Save gestures and their assigned actions to gestures.json"""
    with open(GESTURE_FILE, "w") as file:
        json.dump(gesture_mappings, file, indent=4)

# Function to execute an action
def perform_action(action):
    """Executes the saved action when a recognized gesture is detected."""
    if "youtube" in action.lower():
        query = urllib.parse.quote(action.replace("open youtube", "").strip())
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
    elif "open" in action.lower():
        app_name = action.split("open", 1)[-1].strip()
        try:
            subprocess.Popen([app_name], shell=True)
        except FileNotFoundError:
            st.warning(f"Could not find application '{app_name}'")
    else:
        st.write(f"Performing action: {action}")

# Detect hand gestures using Gemini AI
def detect_gesture(frame):
    """Uses Gemini AI to detect gestures in the given frame."""
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode()

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Identify the hand gesture and return a single short label.",
            {"mime_type": "image/jpeg", "data": img_base64}
        ])
        return response.text.strip().lower()
    except Exception as e:
        st.error(f"Error detecting gesture: {e}")
    return None

# Streamlit UI Elements
st.subheader("🎥 Webcam Controls")
start_camera = st.checkbox("Start Camera")
recording_mode = st.checkbox("Start Recording")
quit_camera = st.button("Quit Camera")

if start_camera and not quit_camera:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            detected_gesture = detect_gesture(frame)

            if detected_gesture:
                if recording_mode:
                    if detected_gesture not in gesture_mappings:
                        # Prompt user for action
                        action = st.text_input(f"✍ Enter action for gesture '{detected_gesture}':", key=f"action_{detected_gesture}")
                        if action:
                            # Save gesture-action mapping
                            gesture_mappings[detected_gesture] = action
                            save_gestures()  
                            st.success(f"✅ Gesture '{detected_gesture}' saved with action '{action}'")

                else:
                    if detected_gesture in gesture_mappings:
                        perform_action(gesture_mappings[detected_gesture])

        stframe.image(frame, channels="RGB")

        if quit_camera:
            break

    cap.release()
