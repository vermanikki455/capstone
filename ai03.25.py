# streamlit_app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose()
        self.counter = 0
        self.stage = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    self.stage = "down"
                if angle < 40 and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1

                cv2.putText(image, f'Reps: {self.counter}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f'Stage: {self.stage}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f'Angle: {int(angle)}', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except:
                pass

            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit UI
st.title("🏋️ Curl Counter with MediaPipe + Streamlit")
webrtc_streamer(
    key="pose-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PoseProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
