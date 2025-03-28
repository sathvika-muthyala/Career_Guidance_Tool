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
from io import StringIO
import requests
import plotly.express as px
import pandas as pd
import streamlit as st
import itertools
from dotenv import load_dotenv
import os
import nltk
import spacy

nltk.download('punkt')
spacy.cli.download("en_core_web_md")
nlp = spacy.load("en_core_web_md")

load_dotenv()  # Load the .env file
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HF_API_KEY = os.getenv("HF_API_KEY")

print(f"HF_API_KEY: {HF_API_KEY}")  # This should print the API key if loaded correctly

if not HF_API_KEY:
    raise ValueError("HF_API_KEY environment variable not set!")
    
HEADERS = {"Authorization": "Bearer " + HF_API_KEY}


def chat_with_ai(user_input):
    """Calls Hugging Face API for more accurate responses."""
    payload = {"inputs": user_input}
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    return "Sorry, I couldn't process your request."

# AI Chatbot UI in Sidebar
st.sidebar.title("AI Career Chatbot")
st.sidebar.write("Ask me anything about job roles, skills, salaries!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
user_input = st.sidebar.text_input("Ask me about career guidance...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate AI response using Hugging Face API
    response = chat_with_ai(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display user input & AI response
    st.sidebar.text_area("You:", user_input, height=70)
    st.sidebar.text_area("AI:", response, height=120)

# Text Preprocessing
def preprocess_text(text):
    """Tokenizes, removes extra spaces, and processes text for skill matching using spaCy."""
    # Normalize text: lowercase and remove extra spaces
    cleaned_text = " ".join(text.lower().split())
    doc = nlp(cleaned_text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

def preprocess_skills_list(text):
    """Splits a comma-separated string of skills and removes extra spaces."""
    return [skill.strip().lower() for skill in text.split(',') if skill.strip()]

# Data Collection & Preprocessing 
@st.cache_data
def load_data():
    """Loads job roles and required skills from a dataset."""
    return pd.read_csv("./job_skills_data-3.csv")  # Modify with your actual path

df = load_data()

# Create a processed skills list column from the Required_Skills field
df["skills_list"] = df["Required_Skills"].apply(preprocess_skills_list)
# (Optional) Create a full text version for semantic matching by joining the list
df["processed_skills"] = df["skills_list"].apply(lambda lst: " ".join(lst))

# Enhanced EDA Functions 
def display_eda_summary(df):
    """Display data overview, structure, and summary statistics."""
    st.subheader("Enhanced EDA - Data Summary")
    st.write("**Data Overview (first 5 rows):**")
    st.dataframe(df.head())
    
    st.write("**Data Structure:**")
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.write("**Summary Statistics:**")
    st.dataframe(df.describe())

def plot_missing_values(df):
    """Plot a heatmap of missing values."""
    st.subheader("Missing Values Analysis")
    missing_values = df.isnull().sum()
    st.write(f"Missing values per column:\n{missing_values}")
    if missing_values.any():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No missing values detected.")

def plot_correlation_matrix(df):
    """Plot a correlation matrix for numeric columns."""
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation analysis.")

def plot_salary_distribution(df):
    """Plot salary distribution by job role using a box plot."""
    st.subheader("Salary Distribution by Job Role")
    if 'Total_Compensation' in df.columns and 'Job_Title' in df.columns:
        fig = px.box(df, x="Job_Title", y="Total_Compensation", title="Salary Distribution by Job Role")
        st.plotly_chart(fig)
    else:
        st.warning("Required columns for salary distribution not found.")


def plot_job_demand_trends(df):
    """Plot monthly job postings trends while ensuring all selected jobs are plotted."""
    st.subheader("📈 Job Demand Trends")

    if 'Posting_Date' in df.columns and 'Job_Title' in df.columns:
        # Convert Posting Date to datetime format
        df['Posting_Date'] = pd.to_datetime(df['Posting_Date'], errors='coerce')
        df.dropna(subset=['Posting_Date'], inplace=True)

        # Extract Month for grouping
        df['Month'] = df['Posting_Date'].dt.to_period('M').astype(str)

        # Group by Month & Job Title for trend analysis
        postings = df.groupby(['Month', 'Job_Title']).size().reset_index(name='Counts')

        # Ensure all job titles appear in every month, even if they have zero postings
        unique_months = postings['Month'].unique()
        unique_titles = postings['Job_Title'].unique()
        all_combinations = pd.DataFrame(itertools.product(unique_months, unique_titles), columns=['Month', 'Job_Title'])
        postings = all_combinations.merge(postings, on=['Month', 'Job_Title'], how='left').fillna(0)

        # **🔹 Debugging: Print unique job titles**
        st.write("Available Job Titles:", postings['Job_Title'].unique())

        # **🔹 Add a dynamic select box for users to pick jobs**
        all_jobs = postings['Job_Title'].unique()
        selected_jobs = st.multiselect("Select Job Roles to Display:", all_jobs, default=all_jobs[:10])

        # Filter data based on user-selected job titles
        filtered_data = postings[postings['Job_Title'].isin(selected_jobs)]

        st.write("Filtered Data Sample:", filtered_data.head())

        # Check if any selected jobs have no data
        missing_jobs = set(selected_jobs) - set(filtered_data['Job_Title'].unique())
        if missing_jobs:
            st.warning(f"The following job titles had no data and were not plotted: {missing_jobs}")

        # Plot the updated job demand trend with more roles
        fig = px.line(
            filtered_data,
            x="Month",
            y="Counts",
            color="Job_Title",
            title="📊 Updated Monthly Job Postings Trend",
            labels={"Counts": "Number of Job Postings", "Month": "Month"},
        )

        # Improve layout: Make the legend scrollable & ensure all job titles are included
        fig.update_layout(
            showlegend=True,
            legend=dict(
                title="Job Titles",
                yanchor="top",
                y=1.02,
                xanchor="left",
                x=1.1,
                font=dict(size=10),
                traceorder="normal"
            )
        )

        st.plotly_chart(fig)

    else:
        st.warning("⚠️ Required columns for job demand trends not found.")



def plot_geographic_distribution(df):
    """Plot job postings by location using a bar chart."""
    st.subheader("Geographic Distribution of Job Postings")
    if 'Location' in df.columns:
        location_counts = df['Location'].value_counts().reset_index()
        location_counts.columns = ['Location', 'Counts']
        fig = px.bar(location_counts, x='Location', y='Counts', title="Job Postings by Location")
        st.plotly_chart(fig)
    else:
        st.warning("Location column not found.")

# Enhanced Skill Matching 
def get_skill_match_score(user_skills, job_skills, user_experience_level="Intermediate"):
    """Calculates a match score using spaCy embeddings and adjusts it based on experience level."""
    user_vector = nlp(user_skills).vector
    job_vector = nlp(job_skills).vector
    cosine_sim = cosine_similarity([user_vector], [job_vector])[0][0]
    # Adjust match score based on experience level
    experience_multiplier = {"Beginner": 0.5, "Intermediate": 0.75, "Advanced": 1.0}
    experience_factor = experience_multiplier.get(user_experience_level, 1.0)
    return cosine_sim * experience_factor

# Job Market Trends from Dataset 
def fetch_job_market_trends():
    """Fetches job market trends based on required skills frequency."""
    skill_counts = df["Required_Skills"].str.split(',').explode().str.strip().value_counts()
    return skill_counts.to_dict()

# Company-Specific Skill Recommendations
def get_company_specific_skills(company, job_role):
    """
    Fetches required skills for the selected job role at the specified company.
    This filters the dataset by both company and job role.
    """
    subset = df[(df["Company"].str.lower() == company.lower()) & (df["Job_Title"] == job_role)]
    if not subset.empty:
        skills = set()
        for row in subset["skills_list"]:
            skills.update(row)
        return sorted(skills)
    else:
        return []

# Salary Prediction Feature
def predict_salary(skill_match_score, experience_level, job_role):
    """Predicts estimated salary based on skill match score, experience level, and job role."""
    job_data = df[df["Job_Title"] == job_role].iloc[0]
    base_salary = job_data["Total_Compensation"]
    experience_multiplier = {"Beginner": 1.0, "Intermediate": 1.5, "Advanced": 2.0}
    predicted_salary = base_salary * experience_multiplier.get(experience_level, 1.0) * skill_match_score
    return round(predicted_salary, 2)

# Streamlit UI 
st.title("AI-Powered Career Guidance Tool")
st.write("Enter your **current skills** and get **AI-powered recommendations** for your dream job!")

# Enhanced EDA Section
if st.checkbox("Show Enhanced EDA Analysis"):
    display_eda_summary(df)
    plot_missing_values(df)
    plot_correlation_matrix(df)
    plot_salary_distribution(df)
    plot_job_demand_trends(df)
    plot_geographic_distribution(df)

# User Inputs
user_skills = st.text_area("Enter your current skills (comma-separated)", "").lower()
job_role = st.selectbox("Select your dream job role", df["Job_Title"].unique())
user_experience_level = st.selectbox("Select your skill experience level", ["Beginner", "Intermediate", "Advanced"])
companies_list = sorted(df["Company"].dropna().unique())
dream_company = st.selectbox("Select your dream company", companies_list)

if st.button("Get AI Recommendations"):
    progress_bar = st.progress(0)
    if user_skills:
        user_skills_list = preprocess_skills_list(user_skills)
        job_data = df[df["Job_Title"] == job_role].iloc[0]
        job_skills_list = job_data["skills_list"]
        user_skills_text = " ".join(user_skills_list)
        job_skills_text = " ".join(job_skills_list)
        
        # Updated to use the user-selected experience level
        skill_match_score = get_skill_match_score(user_skills_text, job_skills_text, user_experience_level)
        
        missing_skills = set(job_skills_list) - set(user_skills_list)
        
        for i in range(100):
            progress_bar.progress(i + 1)
            
        st.subheader("AI Skill Gap Analysis")
        st.write(f"**Match Score:** {round(skill_match_score * 100, 2)}%")
        if missing_skills:
            st.write(f"**Missing Skills:** {', '.join(sorted(missing_skills))}")
        else:
            st.success("You have all the required skills! 🎉")
        
        st.subheader("AI-Recommended Learning Resources")
        st.write(f"Suggested Course: {job_data['Learning Resources']}")
        
        estimated_salary = predict_salary(skill_match_score, user_experience_level, job_role)
        st.subheader("Estimated Salary")
        st.write(f"Expected Salary: **${estimated_salary}** per year")
        
        company_skills = get_company_specific_skills(dream_company, job_role)
        st.subheader(f"🏢 Skills Needed for {dream_company} ({job_role})")
        if company_skills:
            st.write(f"Recommended Skills: {', '.join(company_skills)}")
        else:
            st.info(f"No specific skill recommendations found for {dream_company} for the role of {job_role}.")

# Job Alerts Feature
st.sidebar.subheader("📢 Get Job Alerts!")
email = st.sidebar.text_input("Enter your email to receive job alerts")
if st.sidebar.button("Subscribe"):
    if email:
        st.sidebar.success("You have been subscribed to job alerts!")
    else:
        st.sidebar.warning("Please enter a valid email address.")

# Job Market Trends
st.subheader("Job Market Trends")
job_trends = fetch_job_market_trends()
fig = px.bar(x=list(job_trends.keys()), y=list(job_trends.values()),
             labels={'x': "Skills", 'y': "Market Demand"}, title="Top Skills in Demand")
st.plotly_chart(fig)

# Skill Progress Tracker
st.sidebar.title("Skill Progress Tracker")
if job_role in df["Job_Title"].values:
    job_data = df[df["Job_Title"] == job_role].iloc[0]
    job_skills_list = job_data["skills_list"]
else:
    job_skills_list = []

if "completed_skills" not in st.session_state:
    st.session_state.completed_skills = set()
tracked_skills_input = st.sidebar.text_area(
    "Mark completed skills (comma-separated)", 
    value=", ".join(st.session_state.completed_skills) if st.session_state.completed_skills else ""
)
tracked_skills_list = preprocess_skills_list(tracked_skills_input)
if st.sidebar.button("Update Progress"):
    if tracked_skills_list:
        st.session_state.completed_skills = set(tracked_skills_list)
        st.sidebar.success("Progress updated!")
    else:
        st.sidebar.warning("Please enter at least one skill!")
if job_skills_list:
    completed_skills = st.session_state.completed_skills
    acquired_skills = set(job_skills_list).intersection(completed_skills)
    missing_skills_progress = set(job_skills_list) - completed_skills
    progress = len(acquired_skills) / len(job_skills_list)
    st.subheader("Your Skill Progress")
    st.write(f"**Acquired Skills:** {', '.join(sorted(acquired_skills)) if acquired_skills else 'None'}")
    st.write(f"**Remaining Skills:** {', '.join(sorted(missing_skills_progress)) if missing_skills_progress else 'None'}")
    st.progress(progress)
    if progress == 1:
        st.success("🎉 Congratulations! You've completed all the required skills for this job role.")
    else:
        st.info(f"You're {round(progress * 100, 2)}% there! Keep learning.")
