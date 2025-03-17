import streamlit as st
import pandas as pd
import requests
import nltk
import spacy
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nlp = spacy.load("en_core_web_md")

# -------------------- 1️⃣ Data Collection & Preprocessing --------------------
@st.cache_data
def load_data():
    """Loads job roles and required skills from a dataset."""
    return pd.read_csv("/Users/sat/Career/skills-analysis-system/job_skills_data-3.csv")  # Modify with your actual path

df = load_data()

# EDA: Basic Information & Summary
def display_eda_summary(df):
    """Display basic statistics and data structure."""
    st.subheader("📊 EDA - Data Summary")
    st.write("Data Overview:")
    st.write(df.head())
    
    st.write("Data Types and Missing Values:")
    st.write(df.info())
    
    st.write("Summary Statistics:")
    st.write(df.describe())
    
# EDA: Visualizing Missing Values
def plot_missing_values(df):
    """Plot missing values heatmap."""
    st.subheader("🔴 Missing Values Analysis")
    missing_values = df.isnull().sum()
    st.write(f"Missing values per column:\n{missing_values}")
    if missing_values.any():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
        st.pyplot(fig)

# EDA: Correlation Matrix
def plot_correlation_matrix(df):
    """Plot correlation heatmap for numeric columns."""
    st.subheader("🔍 Correlation Matrix")
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])
    
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation analysis.")

# -------------------- 2️⃣ Enhanced Skill Matching --------------------
def preprocess_text(text):
    """Tokenizes and processes text for skill matching using spaCy."""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

df["processed_skills"] = df["Required_Skills"].apply(preprocess_text)

def get_skill_match_score(user_skills, job_skills, user_experience_level="Intermediate"):
    """Calculates a match score using spaCy embeddings."""
    user_vector = nlp(user_skills).vector
    job_vector = nlp(job_skills).vector
    cosine_sim = cosine_similarity([user_vector], [job_vector])[0][0]

    # Adjust the match score based on the user's experience level
    experience_multiplier = {"Beginner": 0.5, "Intermediate": 0.75, "Advanced": 1.0}
    experience_factor = experience_multiplier.get(user_experience_level, 1.0)

    return cosine_sim * experience_factor

# -------------------- 3️⃣ Job Market Trends from Dataset --------------------
def fetch_job_market_trends():
    """Fetches job market trends from the dataset."""
    skill_counts = df["Required_Skills"].str.split(',').explode().str.strip().value_counts()
    return skill_counts.to_dict()

# -------------------- 4️⃣ Company-Specific Skill Recommendations --------------------
def get_company_specific_skills(company):
    """Fetches required skills for a specific company (mock API)."""
    company_skills = {
        "Google": ["Python", "Machine Learning", "TensorFlow", "Data Structures"],
        "Amazon": ["Java", "AWS", "Big Data", "Distributed Systems"],
        "Microsoft": ["C#", "Azure", "SQL", "Data Engineering"]
    }
    return company_skills.get(company, [])

# -------------------- 5️⃣ Salary Prediction Feature --------------------
def predict_salary(skill_match_score, experience_level, job_role):
    """Predicts estimated salary based on skill match score, experience, and job role."""
    
    # Get the base salary from the dataset based on the job role
    job_data = df[df["Job_Title"] == job_role].iloc[0]  # Assuming 'Job_Title' column contains the job roles
    base_salary = job_data["Total_Compensation"]  # Assuming 'Estimated_Salary' column contains the estimated salary data
    
    # Define experience multipliers
    experience_multiplier = {"Beginner": 1.0, "Intermediate": 1.5, "Advanced": 2.0}
    
    # Calculate the predicted salary
    predicted_salary = base_salary * experience_multiplier.get(experience_level, 1.0) * skill_match_score
    return round(predicted_salary, 2)


# -------------------- 6️⃣ Streamlit UI --------------------
st.title("🚀 AI-Powered Career Guidance Tool")
st.write("Enter your **current skills** and get **AI-powered recommendations** for your dream job!")

# EDA Section
if st.checkbox("Show EDA Analysis"):
    display_eda_summary(df)
    plot_missing_values(df)
    plot_correlation_matrix(df)

# User Inputs
user_skills = st.text_area("Enter your current skills (comma-separated)", "").lower()
job_role = st.selectbox("Select your dream job role", df["Job_Title"].unique())
user_experience_level = st.selectbox("Select your skill experience level", ["Beginner", "Intermediate", "Advanced"])
dream_company = st.selectbox("Select your dream company", ["Google", "Amazon", "Microsoft", "Other"])

# Initialize missing_skills outside the button block
missing_skills = set()

# Button to Get Recommendations
if st.button("Get AI Recommendations"):
    progress_bar = st.progress(0)
    if user_skills:
        processed_user_skills = preprocess_text(user_skills)
        job_data = df[df["Job_Title"] == job_role].iloc[0]
        job_skills = job_data["processed_skills"]
        skill_match_score = get_skill_match_score(processed_user_skills, job_skills)
        missing_skills = set(word_tokenize(job_skills)) - set(word_tokenize(processed_user_skills))

        for i in range(100):
            progress_bar.progress(i + 1)

        st.subheader("🔍 AI Skill Gap Analysis")
        st.write(f"**Match Score:** {round(skill_match_score * 100, 2)}%")
        if missing_skills:
            st.write(f"**Missing Skills:** {', '.join(missing_skills)}")
        else:
            st.success("You have all the required skills! 🎉")
        
        st.subheader("📚 AI-Recommended Learning Resources")
        st.write(f"Suggested Course: {job_data['Learning Resources']}")
        
        # Salary Prediction
        estimated_salary = predict_salary(skill_match_score, user_experience_level, job_role)  # Pass job_role here
        st.subheader("💰 Estimated Salary")
        st.write(f"Expected Salary: **${estimated_salary}** per year")
        
        # Company-Specific Recommendations
        if dream_company != "Other":
            company_skills = get_company_specific_skills(dream_company)
            st.subheader(f"🏢 Skills Needed for {dream_company}")
            st.write(f"Recommended Skills: {', '.join(company_skills)}")

# ----------------- 📬 Job Alerts Feature -----------------
st.sidebar.subheader("📢 Get Job Alerts!")
email = st.sidebar.text_input("Enter your email to receive job alerts")
if st.sidebar.button("Subscribe"):
    if email:
        st.sidebar.success("You have been subscribed to job alerts!")
    else:
        st.sidebar.warning("Please enter a valid email address.")

# ----------------- 📈 Job Market Trends -----------------
st.subheader("📊 Job Market Trends")
job_trends = fetch_job_market_trends()
fig = px.bar(x=list(job_trends.keys()), y=list(job_trends.values()),
             labels={'x': "Skills", 'y': "Market Demand"}, title="Top Skills in Demand")
st.plotly_chart(fig)

# ----------------- 🎯 Skill Progress Tracker -----------------
st.sidebar.title("📊 Skill Progress Tracker")
if job_role in df["Job_Title"].values:
    job_data = df[df["Job_Title"] == job_role].iloc[0]
    job_skills = job_data["processed_skills"]
    required_skills_set = set(word_tokenize(job_skills))
else:
    required_skills_set = set()

if "completed_skills" not in st.session_state:
    st.session_state.completed_skills = set()
tracked_skills_input = st.sidebar.text_area(
    "Mark completed skills (comma-separated)", 
    value=", ".join(st.session_state.completed_skills) if st.session_state.completed_skills else ""
)
tracked_skills_input = preprocess_text(tracked_skills_input)
if st.sidebar.button("Update Progress"):
    if tracked_skills_input.strip():
        st.session_state.completed_skills = set(word_tokenize(tracked_skills_input.lower()))
        st.sidebar.success("Progress updated!")
    else:
        st.sidebar.warning("Please enter at least one skill!")

if required_skills_set:
    completed_skills = st.session_state.completed_skills
    acquired_skills = required_skills_set.intersection(completed_skills)
    missing_skills = required_skills_set - completed_skills
    progress = len(acquired_skills) / len(required_skills_set)

    st.subheader("🎯 Your Skill Progress")
    st.write(f"✅ **Acquired Skills:** {', '.join(acquired_skills) if acquired_skills else 'None'}")
    st.write(f"❌ **Remaining Skills:** {', '.join(missing_skills) if missing_skills else 'None'}")
    st.progress(progress)
    if progress == 1:
        st.success("🎉 Congratulations! You've completed all the required skills for this job role.")
    else:
        st.info(f"You're {round(progress * 100, 2)}% there! Keep learning. 🚀")
