import os
import cv2
import json
import base64
import streamlit as st
import mediapipe as mp
import google.generativeai as genai
import difflib

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBJ_mi6GIj3ZpRIFX8rPZj0iXFxYW4sntI")  # Replace with actual key
if not API_KEY:
    st.error("API Key is missing! Set GEMINI_API_KEY as an environment variable.")
    st.stop()

genai.configure(api_key=API_KEY)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture file path
GESTURE_FILE = "gestures.json"

# Load saved gestures from JSON
def load_gestures():
    if os.path.exists(GESTURE_FILE):
        try:
            with open(GESTURE_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            st.warning("⚠️ Corrupt gestures.json file. Resetting...")
            return {}
    return {}

# Save gestures to JSON
def save_gestures():
    try:
        with open(GESTURE_FILE, "w", encoding="utf-8") as file:
            json.dump(st.session_state.gesture_mappings, file, indent=4)
        st.success("✅ Gesture saved successfully!")
    except Exception as e:
        st.error(f"Error saving gesture: {e}")

# Initialize gestures in session state
if "gesture_mappings" not in st.session_state:
    st.session_state.gesture_mappings = load_gestures()

# Update stored gestures from JSON when recording mode is turned off
def refresh_gestures():
    st.session_state.gesture_mappings = load_gestures()

# Extract gesture label from API response
def extract_gesture_label(response_text):
    return response_text.split("\n")[0].strip().lower().replace("**", "").replace("*", "").replace("'", "").replace(".", "")

# Generate system command using AI
def get_system_command(action):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Generate the correct system command for Windows for this action: {action}. Only return the command.")
        return response.text.strip() if response.text else None
    except Exception as e:
        st.error(f"Error fetching system command: {e}")
        return None

# Perform system action
def perform_action(action):
    system_command = get_system_command(action)
    if system_command:
        os.system(system_command)
        st.success(f"Executed: {system_command}")
    else:
        st.warning("No valid command generated.")

# Capture gesture using Gemini API
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
        return extract_gesture_label(response.text.strip()) if response.text else None
    except Exception as e:
        st.error(f"Error fetching gesture from AI: {e}")
        return None

# Find closest gesture match
def find_closest_gesture(gesture_label):
    closest_match = difflib.get_close_matches(gesture_label, st.session_state.gesture_mappings.keys(), n=1, cutoff=0.7)
    return closest_match[0] if closest_match else None

# Streamlit UI
st.title("🖐️ Gesture-Interactive Chatbot")
st.write("Use live webcam to detect hand gestures and trigger system actions.")

cap = cv2.VideoCapture(0)
stframe = st.empty()
recording_mode = st.checkbox("Enable Recording Mode", key="record_mode")

# Refresh gestures when recording mode is turned off
if not recording_mode:
    refresh_gestures()

# Quit button (Fixed duplicate key issue)
if st.button("Quit", key="quit_app"):
    cap.release()
    st.write("Closed camera stream.")
    st.stop()

# Display stored gestures
st.subheader("Stored Gestures and Actions")
if st.session_state.gesture_mappings:
    for gesture, action in st.session_state.gesture_mappings.items():
        st.write(f"**{gesture}:** {action}")
else:
    st.write("No gestures stored yet.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Couldn't capture frame")
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        gesture_name = capture_gesture(frame)
        if gesture_name:
            st.write(f"**Detected Gesture:** {gesture_name}")
            matched_gesture = find_closest_gesture(gesture_name)
            
            if matched_gesture:
                perform_action(st.session_state.gesture_mappings[matched_gesture])
            elif recording_mode:
                # Ensure unique input key
                input_key = f"input_{gesture_name}_{hash(gesture_name)}"
                action = st.text_input(f"Enter the action for '{gesture_name}':", key=input_key)

                # Ensure unique save button key
                if st.button(f"Save Gesture {gesture_name}", key=f"save_{gesture_name}_{hash(gesture_name)}"):
                    if action:
                        system_command = get_system_command(action)
                        if system_command:
                            # ✅ Save gesture dynamically
                            st.session_state.gesture_mappings[gesture_name] = system_command
                            save_gestures()  # ✅ Save to JSON
                            refresh_gestures()  # ✅ Refresh immediately
                            st.success(f"Gesture '{gesture_name}' saved! → Command: '{system_command}'")
                        else:
                            st.warning("Failed to generate a command.")
                    else:
                        st.warning("Please enter an action.")

    stframe.image(frame, channels="BGR")
    
cap.release()
st.write("Closed camera stream.")
