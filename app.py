import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components
import joblib

# Set page config (title and icon) - must be first Streamlit command
st.set_page_config(
    page_title="AI Hiring Plan Generator",
    page_icon="🤖",  # You can also use a URL to a custom icon
    layout="wide"
)

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing in environment variables.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Load trained ML models
model_path = "ai_hiring_models.joblib"
try:
    loaded_models = joblib.load(model_path)
except Exception as e:
    st.warning(f"⚠️ ML models not loaded: {e}")
    loaded_models = None

# Inject custom CSS animations + hover + 3D effects
st.markdown("""
<style>
    html, body {
        background-color: #f7f9fc;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextArea textarea {
        transition: all 0.3s ease;
        border-radius: 8px !important;
        border: 1px solid #ccc;
        box-shadow: 0 0 5px rgba(0,0,0,0.05);
    }
    .stTextArea textarea:hover {
        border-color: #4b9cd3 !important;
        box-shadow: 0 0 15px rgba(75, 156, 211, 0.4);
        transform: scale(1.01);
    }
    .stButton > button {
        transition: all 0.3s ease-in-out;
        border-radius: 6px;
        font-weight: bold;
        background: linear-gradient(90deg, #4b9cd3, #1f77b4);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #1f77b4, #0d5a8c);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        color: white !important;
    }
    .copy-btn {
        background-color: #1f77b4;
        color: white;
        padding: 8px 14px;
        border: none;
        border-radius: 6px;
        font-weight: bold;
        cursor: pointer;
    }
    .copy-btn:hover {
        background-color: #0d5a8c;
    }
</style>
""", unsafe_allow_html=True)

# Prompt template
prompt_template = """
Generate a detailed AI/ML Hiring Plan based on the following information.

**Company Information:**
{company_information}

**Position Details:**
{position_details}

**Project Requirements:**
{project_requirements}

**Team Dynamics:**
{team_dynamics}

**Recruitment Goals:**
{recruitment_goals}

Please generate a hiring plan with the following sections:

**1. Executive Summary:**
Provide a brief overview of the role and the hiring need.

**2. Position Overview:**
Detail the Job Title, reporting structure, and key responsibilities.

**3. Required Skills and Qualifications:**
List the necessary technical skills, soft skills, and any preferred qualifications.

**4. Project Context and Impact:**
Explain the specific AI/ML projects the hire will work on and their expected impact.

**5. Team Integration and Dynamics:**
Describe how the new hire will fit into the existing team structure and collaboration style.

**6. Recruitment Strategy and Process:**
Outline the recruitment timeline, stages, and assessment methods.

**7. Goals and Success Metrics for the Hire:**
Define the key objectives and how the success of the new hire will be measured.

Ensure the plan is detailed, well-structured, and written in a professional tone. Use markdown for clear section headers.
"""

# Streamlit UI
st.title("🤖 AI Hiring Plan Generator")
st.markdown("Fill in the details below to generate a complete hiring plan for your AI/ML role.")

with st.form("hiring_plan_form"):
    company_information = st.text_area("📌 Company Information", placeholder="e.g., HealthTech startup, 50 employees, remote-first culture")
    position_details = st.text_area("💼 Position Details", placeholder="e.g., ML Engineer, Mid-level, reports to Head of AI")
    project_requirements = st.text_area("🛠️ Project Requirements", placeholder="e.g., NLP chatbot using Hugging Face, TensorFlow")
    team_dynamics = st.text_area("🤝 Team Dynamics", placeholder="e.g., 2 Data Scientists, Agile cross-functional team")
    recruitment_goals = st.text_area("🎯 Recruitment Goals", placeholder="e.g., 3 months, ₹15L, 40% diversity target")
    
    submitted = st.form_submit_button("🚀 Generate Hiring Plan")

response_text = ""

if submitted:
    if not all([company_information, position_details, project_requirements, team_dynamics, recruitment_goals]):
        st.warning("⚠️ Please fill in all the fields above.")
    else:
        with st.spinner("⏳ Generating your hiring plan..."):
            prompt = prompt_template.format(
                company_information=company_information,
                position_details=position_details,
                project_requirements=project_requirements,
                team_dynamics=team_dynamics,
                recruitment_goals=recruitment_goals
            )
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                if not response.text:
                    st.error("The model didn't return any text. Please try again.")
                else:
                    response_text = response.text + "\n\nThis plan provides a framework for a successful and efficient recruitment process. Regular review and adjustment based on progress and market conditions will be essential."
                    st.markdown("---")
                    st.subheader("📄 Generated AI Hiring Plan")
                    st.markdown(response_text)

                    # Predictions from ML Model (basic logic)
                    if loaded_models:
                        st.markdown("---")
                        st.markdown("### 🔮 AI-Powered Predictions")
                        user_input_vector = {
                            "Company_Info": company_information,
                            "Position": position_details,
                            "Project": project_requirements,
                            "Team": team_dynamics,
                            "Recruitment": recruitment_goals
                        }
                        # Create dummy input based on the length of concatenated inputs
                        input_length = len(" ".join(user_input_vector.values()))
                        dummy_input = [[input_length] * next(iter(loaded_models.values()))["RandomForest"].n_features_in_]

                        for target, model_bundle in loaded_models.items():
                            rf_model = model_bundle.get("RandomForest")
                            le = model_bundle.get("LabelEncoder")
                            if rf_model and hasattr(rf_model, 'n_features_in_'):
                                pred = rf_model.predict(dummy_input)
                                if le:
                                    try:
                                        pred_label = le.inverse_transform(pred)
                                    except:
                                        pred_label = pred
                                else:
                                    pred_label = pred
                                st.markdown(f"**{target}** ➤ `{pred_label[0]}`")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# If response_text is populated, show editable output and copy/download features
if response_text:
    edited = st.text_area("📝 Edit Your Hiring Plan (optional)", value=response_text, height=400)

    components.html(f'''
    <button class="copy-btn" onclick="navigator.clipboard.writeText(`{edited.replace("`", "\\`").replace("\\", "\\\\")}`)">📋 Copy to Clipboard</button>
    ''', height=50)

    st.download_button(
        label="📥 Download as .txt",
        data=edited,
        file_name="ai_hiring_plan.txt",
        mime="text/plain"
    )