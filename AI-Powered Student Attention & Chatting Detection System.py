import streamlit as st
import cv2
import numpy as np
import torch
import time
import pandas as pd
import mss
from ultralytics import YOLO
import os
from scipy.spatial import distance as dist
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov8n.pt").to(device)
yolo_model.fuse()

st.title("🎓 AI-Powered Student Attention & Chatting Detection")

start_button = st.button("Start Analysis")
stop_button = st.button("Stop Analysis")

if 'is_analysis_running' not in st.session_state:
    st.session_state.is_analysis_running = False
if 'student_data' not in st.session_state:
    st.session_state.student_data = {}

def mouth_aspect_ratio(mouth):
    """Calculate the Mouth Aspect Ratio (MAR) to detect if the student is speaking."""
    A = dist.euclidean(mouth[2], mouth[10])  
    B = dist.euclidean(mouth[4], mouth[8])   
    C = dist.euclidean(mouth[0], mouth[6])   
    mar = (A + B) / (2.0 * C)
    return mar

def analyze_face(face_img, student_id, face_box, frame_width, frame_height):
    try:
        face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_width = face_box[2] - face_box[0]
        face_height = face_box[3] - face_box[1]

        student = st.session_state.student_data.setdefault(student_id, {
            "warnings": 0, "attentive": 0, "chatting": 0, "last_status": "", "start_time": time.time()
        })
        center_x = (face_box[0] + face_box[2]) // 2
        center_y = (face_box[1] + face_box[3]) // 2
        attention_region_x = (frame_width // 4, 3 * frame_width // 4)
        attention_region_y = (frame_height // 4, 3 * frame_height // 4)

        aspect_ratio = face_width / face_height
        looking_at_screen = (
            0.75 < aspect_ratio < 1.3 and 
            attention_region_x[0] < center_x < attention_region_x[1] and 
            attention_region_y[0] < center_y < attention_region_y[1]
        )

        mouth_region = face_img_gray[face_height//2:face_height, face_width//4:3*face_width//4]
        _, mouth_thresh = cv2.threshold(mouth_region, 80, 255, cv2.THRESH_BINARY)
        mar = mouth_aspect_ratio(mouth_thresh)

        if mar > 0.5:  
            student["chatting"] += 1
            student["last_status"] = "chatting"
            st.toast(f"💬 {student_id} is chatting!", icon="💬")
        elif looking_at_screen:
            student["attentive"] += 1
            student["last_status"] = "attentive"
            st.toast(f"✅ {student_id} is attentive and looking straight at the screen.", icon="✅")
        else:
            student["chatting"] += 1
            student["last_status"] = "chatting"
            st.toast(f"🔄 {student_id} is turned away!", icon="❗")

    except Exception as e:
        st.toast(f"⚠ Error: {str(e)}", icon="❌")

if start_button:
    st.session_state.is_analysis_running = True
if stop_button:
    st.session_state.is_analysis_running = False
    
    if st.session_state.student_data:
        report_data = []
        for student_id, data in st.session_state.student_data.items():
            total_time = round(time.time() - data["start_time"], 2)
            report_data.append([
                student_id, data["warnings"], data["chatting"], data["attentive"], total_time
            ])
        df = pd.DataFrame(report_data, columns=[
            "Student ID", "Distraction Warnings", "Chatting", "Attentive", "Total Time (s)"
        ])
        st.write("### 📊 Student Attention & Chatting Report")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Report", csv, "student_report.csv", "text/csv")

if st.session_state.is_analysis_running:
    with mss.mss() as sct:
        while st.session_state.is_analysis_running:
            try:
                screen_img = np.array(sct.grab(sct.monitors[1]), dtype=np.uint8)
                screen_img = cv2.cvtColor(screen_img, cv2.COLOR_RGB2BGR)
                screen_img_small = cv2.resize(screen_img, (640, 360))
                results = yolo_model(screen_img_small, conf=0.6)

                if len(results[0].boxes.xyxy) == 0:
                    st.toast("📷 Camera is off or no student detected!", icon="🚫")
                    for student in st.session_state.student_data.values():
                        student["warnings"] += 1

                for i, box in enumerate(results[0].boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box[:4])
                    x1, y1, x2, y2 = int(x1*2), int(y1*2), int(x2*2), int(y2*2)
                    face_img = screen_img[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                    analyze_face(face_img, f"Student_{i+1}", (x1, y1, x2, y2), screen_img.shape[1], screen_img.shape[0])

                st.image(screen_img, channels="BGR")

            except Exception as e:
                st.toast(f"⚠ Error: {str(e)}", icon="❌")