import streamlit as st
import fitz  # PyMuPDF for PDFs
import docx
import requests
import json
import cv2
import numpy as np
import torch
import time
import pandas as pd
import mss
from ultralytics import YOLO
from scipy.spatial import distance as dist

GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov8n.pt").to(device)
yolo_model.fuse()

if 'is_analysis_running' not in st.session_state:
    st.session_state.is_analysis_running = False
if 'student_data' not in st.session_state:
    st.session_state.student_data = {}
if 'questions_generated' not in st.session_state:
    st.session_state.questions_generated = False
if 'responses' not in st.session_state:
    st.session_state.responses = []

st.title("🎓 AI-Powered Interview & Attention Monitoring System")

def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def generate_questions(content):
    prompt = f"""Analyze this resume and generate 10 interview questions:
    {content}"""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a professional HR interviewer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "Error generating questions"

def evaluate_answers(responses, questions):
    qa_pairs = "\n".join([f"Q: {q}\nA: {r}" for q, r in zip(questions, responses)])
    prompt = f"""Evaluate these interview responses and provide a score (1-10):
    {qa_pairs}"""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an expert HR evaluator."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "Error evaluating responses"

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

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
        
               looking_at_screen = (
            (frame_width//4 < center_x < 3*frame_width//4) and
            (frame_height//4 < center_y < 3*frame_height//4)
        )

     
        mouth_region = face_img[face_height//2:, :]
        gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        _, mouth_thresh = cv2.threshold(gray_mouth, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mouth_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and looking_at_screen:
            student["attentive"] += 1
            student["last_status"] = "attentive"
        elif contours:
            student["chatting"] += 1
            student["last_status"] = "chatting"
        else:
            student["warnings"] += 1
            student["last_status"] = "distracted"

    except Exception as e:
        st.error(f"Face analysis error: {str(e)}")

def monitor_attention():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while st.session_state.is_analysis_running:
            try:
                
                screen_img = np.array(sct.grab(monitor))
                screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)
                
               
                processing_img = cv2.resize(screen_img, (640, 360))
              
                results = yolo_model(processing_img, conf=0.6)
                

                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    # Scale coordinates back to original size
                    x1 = int(x1 * (screen_img.shape[1] / 640))
                    y1 = int(y1 * (screen_img.shape[0] / 360))
                    x2 = int(x2 * (screen_img.shape[1] / 640))
                    y2 = int(y2 * (screen_img.shape[0] / 360))
                    
                    face_img = screen_img[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                        
                    analyze_face(face_img, "Candidate", (x1, y1, x2, y2), 
                               screen_img.shape[1], screen_img.shape[0])
                
                st.image(screen_img, channels="BGR", caption="Live Monitoring")
            
                time.sleep(0.1)
                
            except Exception as e:
                st.error(f"Monitoring error: {str(e)}")
                break

st.subheader("📤 Upload Resume (PDF/DOCX)")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"], key="resume_uploader")

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension == "pdf":
        resume_content = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        resume_content = extract_text_from_docx(uploaded_file)
    else:
        st.error("Invalid file format")
        st.stop()
    
    if st.button("🧐 Generate Interview Questions"):
        st.session_state.is_analysis_running = True
        with st.spinner("Generating questions..."):
            st.session_state.questions_text = generate_questions(resume_content)
            st.session_state.questions_generated = True

if st.session_state.questions_generated:
    st.subheader("📌 Interview Questions")
    questions = st.session_state.questions_text.split("\n")[:10]  # Get first 10 lines
   
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.session_state.responses = []
        for i, question in enumerate(questions, 1):
            st.session_state.responses.append(st.text_area(f"Q{i}: {question}", key=f"response_{i}"))
        
        if st.button("📊 Evaluate Answers"):
            if all(response.strip() for response in st.session_state.responses):
                with st.spinner("Evaluating..."):
                    evaluation = evaluate_answers(st.session_state.responses, questions)
                st.subheader("🎯 Evaluation Results")
                st.write(evaluation)
            else:
                st.warning("Please answer all questions before evaluation")
    
    with col2:
        st.subheader("👁 Attention Monitoring")
        if st.session_state.is_analysis_running:
            monitor_attention()
        
        if st.button("⏹ Stop Monitoring"):
            st.session_state.is_analysis_running = False
            
            if st.session_state.student_data:
                report_data = []
                for student_id, data in st.session_state.student_data.items():
                    total_time = time.time() - data["start_time"]
                    attention_score = int((data["attentive"] / (data["attentive"] + data["chatting"] + data["warnings"])) * 10)
                    report_data.append([
                        student_id,
                        data["warnings"],
                        data["chatting"],
                        data["attentive"],
                        f"{total_time:.1f}s",
                        f"{attention_score}/10"
                    ])
                
                df = pd.DataFrame(report_data, columns=[
                    "Student", "Distractions", "Chatting", "Attentive", "Duration", "Attention Score"
                ])
                
                st.write("### 📊 Attention Report")
                st.dataframe(df)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Report", csv, "attention_report.csv", "text/csv")

st.markdown("---")
st.markdown("🔹 AI-Powered Interview System")