import os
import pdfplumber
import docx
import re
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
    return text

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error extracting DOCX text: {e}")
        return ""

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def extract_basic_info(text):
    name = text.split('\n')[0]
    email = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phone = re.findall(r"\+?\d[\d\-\(\) ]{7,}\d", text)
    linkedin = re.findall(r"https?://(www\.)?linkedin\.com/in/[A-Za-z0-9\-_]+", text)
    github = re.findall(r"https?://(www\.)?github\.com/[A-Za-z0-9\-_]+", text)

    return {
        "Name": name.strip(),
        "Email": email[0] if email else "",
        "Phone": phone[0] if phone else "",
        "LinkedIn": linkedin[0] if linkedin else "",
        "GitHub": github[0] if github else ""
    }

def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "SKILL", "TECHNOLOGY"]]
    return list(set(skills))

def extract_experience_years(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)\s+of\s+experience', text, re.IGNORECASE)
    return max([int(m[0]) for m in matches], default=0)

def extract_education_details(text):
    degrees = ['Bachelor', 'B.Sc', 'Master', 'M.Sc', 'PhD', 'MBA']
    edu_matches = [d for d in degrees if d.lower() in text.lower()]
    universities = re.findall(r'(?:University|Institute|College|School of [A-Za-z ]+)', text)
    return ", ".join(set(edu_matches)), ", ".join(set(universities))

def extract_certifications(text):
    lines = text.split("\n")
    return [line.strip() for line in lines if any(k in line.lower() for k in ['certificate', 'certified'])]

def extract_projects(text):
    lines = text.split("\n")
    return [line.strip() for line in lines if any(k in line.lower() for k in ['project', 'developed'])][:5]

def keyword_match_score(text, keywords):
    return sum(1 for kw in keywords if kw.lower() in text.lower())

def compute_final_score(similarity, keyword_score, experience_years, weight_config):
    return (
        weight_config["similarity"] * similarity +
        weight_config["keywords"] * (keyword_score / 10) +
        weight_config["experience"] * (experience_years / 10)
    )

st.set_page_config(page_title="AI Resume Filtering", layout="wide")
st.title("🤖 AI-Powered Resume Filtering System")
st.markdown("Upload resumes and let AI filter, extract, and rank the best candidates!")

uploaded_files = st.file_uploader("Upload Resume Files (PDF, DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
job_description = st.text_area("Paste Job Description Here")

keywords_input = st.text_input("Enter required keywords (comma-separated)", value="python, machine learning, sql")
keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]

st.sidebar.header("🔧 Scoring Weights")
weight_config = {
    "similarity": st.sidebar.slider("Semantic Similarity Weight", 0.0, 1.0, 0.6),
    "keywords": st.sidebar.slider("Keyword Match Weight", 0.0, 1.0, 0.3),
    "experience": st.sidebar.slider("Experience Weight", 0.0, 1.0, 0.1),
}
min_score = st.sidebar.slider("Minimum Final Score to Include", 0.0, 1.0, 0.5)

if st.button("Analyze Resumes") and uploaded_files and job_description:
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    results = []

    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.name.endswith(".docx"):
            text = extract_text_from_docx(file)
        else:
            continue

        if not text:
            continue

        preprocessed = preprocess_text(text)

     
        resume_embedding = model.encode(preprocessed, convert_to_tensor=True)
        similarity_score = float(util.pytorch_cos_sim(resume_embedding, job_embedding)[0][0])

     
        basic_info = extract_basic_info(text)
        skills = extract_skills(text)
        experience = extract_experience_years(text)
        edu_degree, edu_university = extract_education_details(text)
        certifications = extract_certifications(text)
        projects = extract_projects(text)
        keyword_score = keyword_match_score(preprocessed, keywords)

        final_score = compute_final_score(similarity_score, keyword_score, experience, weight_config)

        if final_score >= min_score:
            results.append({
                "Name": basic_info["Name"],
                "Email": basic_info["Email"],
                "Phone": basic_info["Phone"],
                "LinkedIn": basic_info["LinkedIn"],
                "GitHub": basic_info["GitHub"],
                "Resume File": file.name,
                "Education": edu_degree,
                "University": edu_university,
                "Certifications": ", ".join(certifications),
                "Projects": ", ".join(projects),
                "Extracted Skills": ", ".join(skills),
                "Years of Experience": experience,
                "Keyword Matches": keyword_score,
                "Match Score": round(similarity_score, 2),
                "Final Score": round(final_score, 2)
            })

    if results:
        st.success(f"✅ {len(results)} resumes matched your criteria!")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.highlight_max(subset=['Final Score'], color='lightgreen'), use_container_width=True)
        st.download_button("📥 Download Results as CSV", results_df.to_csv(index=False).encode("utf-8"), "filtered_resumes.csv", "text/csv")
    else:
        st.warning("⚠ No resumes met the minimum score threshold.")