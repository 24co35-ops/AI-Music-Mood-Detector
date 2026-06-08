import streamlit as st
import cv2
import numpy as np
import requests
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

st.set_page_config(page_title="AI Music Mood Detector", layout="wide")
st.title("AI Music Mood Detector")
st.markdown("Your emotion to playlist mapping")

recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EMOTION_MOOD_MAP = {
    "Happy": "upbeat pop dance",
    "Sad": "chill acoustic lofi",
    "Surprise": "epic cinematic electronic",
    "Fear": "dark ambient suspense",
    "Angry": "heavy metal rock",
    "Disgust": "blues jazz moody",
    "Neutral": "chill indie folk"
}

def search_itunes(query):
    response = requests.get(
        "https://itunes.apple.com/search",
        params={"term": query, "media": "music", "limit": 5},
        timeout=10
    )
    response.raise_for_status()
    return response.json()

camera = st.camera_input("Show your face")

if camera:
    img_array = np.frombuffer(camera.getvalue(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected. Please ensure your face is visible and well-lit.")
    else:
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]

        emotion, scores = recognizer.predict_emotions(face_img, logits=False)
        confidence = float(max(scores))

        if confidence < 0.3:
            st.warning("Emotion detected but confidence is low. Please try again with better lighting.")
        else:
            st.subheader("Detected Emotion")
            st.write(f"{emotion} (confidence: {round(confidence, 2)})")

            mood_query = EMOTION_MOOD_MAP.get(emotion, "chill music")

            try:
                results = search_itunes(mood_query)
                st.subheader("Recommended Tracks")
                for track in results.get("results", []):
                    name = track.get("trackName", "Unknown")
                    artist = track.get("artistName", "Unknown")
                    url = track.get("trackViewUrl", "#")
                    preview = track.get("previewUrl")
                    st.markdown(f"- [{name} - {artist}]({url})")
                    if preview:
                        st.audio(preview, format="audio/mp3")
            except Exception as e:
                st.error(f"Music API error: {type(e).__name__}: {e}")