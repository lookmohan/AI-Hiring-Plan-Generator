# 🤖 AI Hiring Plan Generator

The **AI Hiring Plan Generator** is an interactive Streamlit app that allows companies, HR teams, and tech startups to instantly generate tailored hiring plans for AI/ML roles using Google's Gemini LLM and traditional ML predictions. It combines LLM reasoning and tabular machine learning for smarter, faster hiring decisions.

---

## 🌐 Live Demo

👉 [Click here to try the live app](https://ai-hiring-plan-generator.streamlit.app)

---

## 🚀 Features

- ✨ **AI-generated hiring plans** using Google Gemini 1.5
- 📊 **ML predictions** for estimated hiring timelines and outcomes using trained RandomForest and XGBoost models
- 📝 **Editable output** with clipboard + download options
- 🎨 Smooth UI with hover effects, 3D-style animations and modern form design
- ✅ **No credit card needed** — hosted freely on Streamlit Cloud

---

## 🔍 How It Works

1. **You Fill a Form**  
   Enter your company details, project info, team setup, and recruitment goals.

2. **Gemini Generates a Hiring Plan**  
   Using a prompt template, Google's Gemini LLM returns a full hiring strategy in markdown format.

3. **ML Models Make Predictions**  
   Based on your input, the app also predicts values like estimated cost, hiring time, etc. using RandomForest and XGBClassifier.

4. **You Edit, Copy, or Download**  
   The generated plan is fully editable in the app. You can copy it to clipboard or download as `.txt`.

---

## 📁 Project Structure

```
AI Hiring Plan Generator/
│
├── backend/
│   ├── app.py                # Streamlit app code
│   ├── prompt_template.txt   # LLM prompt structure
│   ├── requirements.txt      # Python dependencies
│
├── ai_hiring_models.joblib   # Trained RandomForest + XGB models
├── ai_hiring_plan_dataset_700_rows.csv  # Training data
├── train_model.py            # ML training script
├── .env                      # Google API key (not committed to GitHub)
└── README.md
```

---

## 💻 How to Run Locally

> Requires Python 3.10+ and a Google API key

1. **Clone the repo**
   ```bash
   git clone https://github.com/lookmohan/AI-Hiring-Plan-Generator.git
   cd AI-Hiring-Plan-Generator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Create `.env` file**
   ```
   GOOGLE_API_KEY=your-google-generativeai-key
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run backend/app.py
   ```

---

## 🧠 ML Model Training

To retrain models using your own dataset:

```bash
python train_model.py
```

Output: `ai_hiring_models.joblib`  
*Make sure it's under 100MB for GitHub and Streamlit compatibility.*

---

## 📦 Deployment (Streamlit Cloud)

Already deployed: ✅  
To deploy your own version:

1. Push your code to GitHub  
2. Add your `GOOGLE_API_KEY` in Streamlit Cloud → App Settings → `Secrets`  
3. Upload the `.joblib` file in “Files” section  
4. Set the main file as:  
   ```
   backend/app.py
   ```

---

## 🛠 Tech Stack

- **Frontend**: Streamlit + HTML/CSS (custom animation)
- **LLM**: Google Gemini via `google-generativeai`
- **ML Backend**: RandomForestClassifier + XGBClassifier
- **ML Tools**: scikit-learn, xgboost, joblib
- **Hosting**: Streamlit Cloud

---

## 📄 License

MIT License © 2025 [@lookmohan](https://github.com/lookmohan)

---

## 👤 Author

Built with ❤️ by **Mohanraj R**  
GitHub: [@lookmohan](https://github.com/lookmohan)  
LinkedIn: [linkedin.com/in/your-profile](https://www.linkedin.com/in/moganraj)

