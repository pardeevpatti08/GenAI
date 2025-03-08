import os
import cv2
import json
import base64
import time
import streamlit as st
import mediapipe as mp
import google.generativeai as genai
import difflib

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBMCoxYy-t6-WB8NICBdAn_GQV5NKZDxYU")  # Replace with actual key
genai.configure(api_key=API_KEY)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture file
GESTURE_FILE = "gestures.json"

# Load saved gestures
if os.path.exists(GESTURE_FILE):
    with open(GESTURE_FILE, "r") as file:
        gesture_mappings = json.load(file)
else:
    gesture_mappings = {}

# Function to extract gesture label from AI response
def extract_gesture_label(response_text):
    label = response_text.split("\n")[0].strip()
    label = label.replace("", "").replace("*", "").replace("'", "").replace(".", "").lower()
    return label

# Function to generate chatbot response
def get_chatbot_reply(gesture):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"If a user makes a '{gesture}' gesture, how should a chatbot respond? Provide a friendly response."
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "I couldn't understand that gesture."
    except Exception as e:
        return f"Error generating response: {e}"

# Function to generate system command using AI
def get_system_command(action):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Given the action '{action}', generate the correct system command for Windows. Only return the command without any explanation."
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else None
    except Exception as e:
        return None

# Function to perform system action
def perform_action(action):
    system_command = get_system_command(action)
    if system_command:
        os.system(system_command)

# Function to capture gesture using Gemini API
def capture_gesture(frame):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode()

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Identify the hand gesture and return a single short label.",
            {"mime_type": "image/jpeg", "data": img_base64}
        ])

        if response.text:
            return extract_gesture_label(response.text.strip())  # Get short name
    except Exception as e:
        return None

# Function to save gestures
def save_gestures():
    with open(GESTURE_FILE, "w") as file:
        json.dump(gesture_mappings, file, indent=4)

# Function to match gestures with saved actions
def find_closest_gesture(gesture_label):
    if not gesture_mappings:
        return None
    closest_match = difflib.get_close_matches(gesture_label, gesture_mappings.keys(), n=1, cutoff=0.7)
    return closest_match[0] if closest_match else None

# Streamlit UI
st.title("🖐 Gesture-Based AI Assistant")

# Sidebar Options
mode = st.sidebar.radio("Select Mode:", ["Gesture Chatbot 🤖", "Gesture-to-System Commands 🔧"])

# Start/Stop Camera Buttons
start_camera = st.sidebar.button("Start Camera", key="start_camera_btn")
stop_camera = st.sidebar.button("Stop Camera", key="stop_camera_btn")

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_window = st.empty()

if start_camera:
    st.sidebar.write("🎥 Camera Started. Show a gesture!")

gesture_processing = False  # Flag to prevent multiple detections at once

while start_camera and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Couldn't capture frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and not gesture_processing:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Recognize gesture
        gesture_name = capture_gesture(frame)
        if gesture_name:
            gesture_processing = True  # Block further detections until response is given
            st.sidebar.write(f"🖐 Detected Gesture: {gesture_name}")

            if mode == "Gesture Chatbot 🤖":
                chatbot_reply = get_chatbot_reply(gesture_name)
                st.write(f"🤖 Chatbot: {chatbot_reply}")

            elif mode == "Gesture-to-System Commands 🔧":
                matched_gesture = find_closest_gesture(gesture_name)
                if matched_gesture:
                    perform_action(gesture_mappings[matched_gesture])
                else:
                    st.warning(f"⚠ Gesture {gesture_name} not recognized. Add a command.")

            time.sleep(3)  # Wait before allowing the next gesture detection
            gesture_processing = False  # Reset flag

    # Display video feed in Streamlit
    frame_window.image(frame, channels="RGB")

    # Stop camera condition
    if stop_camera:
        cap.release()
        cv2.destroyAllWindows()
        st.sidebar.write("⏹ Camera Stopped")
        break

cap.release()
cv2.destroyAllWindows()

# Save new gestures
if mode == "Gesture-to-System Commands 🔧":
    if st.sidebar.button("Add New Gesture", key="add_gesture_btn"):
        gesture_label = st.text_input("Enter the new gesture name:")
        action = st.text_input(f"Enter the action for {gesture_label}:")

        if st.button("Save Gesture", key="save_gesture_btn"):
            if gesture_label and action:
                system_command = get_system_command(action)
                if system_command:
                    gesture_mappings[gesture_label] = system_command
                    save_gestures()
                    st.success(f"✅ Gesture {gesture_label} saved with action {action} → Command: {system_command}")
                else:
                    st.warning("⚠ Failed to generate a command.")

st.sidebar.write("🎯 *Recognized Gestures*")
for gesture, action in gesture_mappings.items():
    st.sidebar.write(f"🔹 {gesture} → {action}")