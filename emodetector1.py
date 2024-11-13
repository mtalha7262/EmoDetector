
import cv2
import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model
import threading
import streamlit as st
import time

# Constants
SAMPLE_RATE = 44100
DURATION = 5
FRAME_SKIP = 3  # Process every 3rd frame for efficiency (adjust as needed)
VIDEO_CAPTURE_DEVICE = 0

# Model paths
FACIAL_MODEL_PATH = r'D:\Emotion detector\emotiondetector.h5'
VOICE_MODEL_PATH = r'D:\Emotion detector\toronto_speech_model.h5'

# Emotion mapping
FACIAL_EMOTION_MAP = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                      4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

VOICE_EMOTION_MAP = {0: 'Happy', 1: 'Angry', 2: 'PS', 3: 'Disgust',
                     4: 'Neutral', 5: 'Sad', 6: 'Fear'}


class EmotionDetector:
    def __init__(self):
        self.facial_model = self.load_model(FACIAL_MODEL_PATH)
        self.voice_model = self.load_model(VOICE_MODEL_PATH)
        self.predicted_voice_emotion = "Waiting for voice..."
        self.predicted_facial_emotion = "Unknown"
        self.video_capture = None  # Initialize video capture object
        self.stop_threads = False  # Flag to stop threads

    def load_model(self, path):
        try:
            model = load_model(path)
            return model
        except FileNotFoundError:
            st.error(f"Model file not found at {path}. Please check the path.")
            raise
        except Exception as e:
            st.error(f"Error loading model from {path}: {e}")
            raise

    def predict_facial_emotion(self, frame):
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (48, 48))
            normalized_frame = resized_frame / 255.0
            input_data = np.expand_dims(np.expand_dims(normalized_frame, -1), 0)
            emotion_prediction = self.facial_model.predict(input_data)
            self.predicted_facial_emotion = FACIAL_EMOTION_MAP[np.argmax(emotion_prediction)]
        except Exception as e:
            st.error(f"Error predicting facial emotion: {e}")

    def predict_voice_emotion(self, recording):
        try:
            audio_features = librosa.feature.mfcc(y=recording.flatten(), sr=SAMPLE_RATE, n_mfcc=13)
            audio_features = np.mean(audio_features.T, axis=0)
            input_data = np.expand_dims(audio_features, axis=0)
            emotion_prediction = self.voice_model.predict(input_data)
            self.predicted_voice_emotion = VOICE_EMOTION_MAP[np.argmax(emotion_prediction)]
        except Exception as e:
            st.error(f"Error predicting voice emotion: {e}")

    def display_emotions_on_frame(self, frame):
        cv2.putText(frame, f'Facial Emotion: {self.predicted_facial_emotion}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Voice Emotion: {self.predicted_voice_emotion}', 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def capture_video(self):
        cv2.ocl.setUseOpenCL(False)
        self.video_capture = cv2.VideoCapture(VIDEO_CAPTURE_DEVICE)
        if not self.video_capture.isOpened():
            st.error(f"Error opening video capture device: {VIDEO_CAPTURE_DEVICE}")
            return

        stframe = st.empty() 
        frame_count = 0
        target_fps = 20  
        frame_time = 1 / target_fps  

        while not self.stop_threads:
            ret, frame = self.video_capture.read()
            if not ret:
                st.error("Error reading from webcam.")
                break

            frame_count += 1

            if frame_count % FRAME_SKIP == 0:
                self.predict_facial_emotion(frame)

            self.display_emotions_on_frame(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

            start_time = time.time()
            delay = max(0, frame_time - (time.time() - start_time))
            time.sleep(delay)

        self.video_capture.release()

    def record_audio(self):
        while not self.stop_threads:
            print("Recording Audio...")
            recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
            sd.wait()  # Wait until recording is finished
            self.predict_voice_emotion(recording)
            print(f"Predicted Voice Emotion: {self.predicted_voice_emotion}")

def main():
    st.title("Real-Time Emotion Detection")
    st.header("This application detects emotions from your facial expressions and voice tone.")
    st.write("")

    emotion_detector = EmotionDetector()

    # UI Buttons
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start")
    with col2:
        stop_button = st.button("Stop")

    if start_button:
        emotion_detector.stop_threads = False  # Reset stop flag
        audio_thread = threading.Thread(target=emotion_detector.record_audio)
        audio_thread.start()
        emotion_detector.capture_video()
        audio_thread.join()

    if stop_button:
        emotion_detector.stop_threads = True

if __name__ == "__main__":
    main()
