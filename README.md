# 🤖 AI Resume Parser & Job Recommender

**Live Demo → [https://ai-resume-parser-9mo4.onrender.com](https://ai-resume-parser-9mo4.onrender.com)**

> ⚠️ Hosted on Render free tier — may take 30–60 seconds to wake up on first visit.

---

## What It Does

Upload any resume (PDF or TXT) and the system will instantly:

- 🗂️ **Categorize** the resume into an industry (Data Science, Healthcare, Finance, etc.)
- 💼 **Recommend** a suitable job role based on resume content
- 👤 **Extract** name, email, phone number
- 🛠️ **Identify** skills mentioned in the resume
- 🎓 **Detect** educational background

---

## Demo

| Upload Screen | Results |
|---|---|
| Upload PDF or TXT resume | Get category, job recommendation, and extracted info instantly |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Model | Random Forest Classifier |
| NLP | TF-IDF Vectorizer, Regex-based extraction |
| Dataset | UpdatedResumeDataSet (962 labeled resumes, 25 categories) |
| Frontend | HTML, CSS |
| Deployment | Render (free tier) |

---

## Model Performance

| Model | Accuracy |
|---|---|
| Resume Categorization | 100% (test set) |
| Job Recommendation | 100% (test set) |

> Trained on 962 resumes across 25 job categories using Random Forest + TF-IDF with 5000 features.

---

## Project Structure

```
AI-Resume-parser/
│
├── APP.py                          # Flask app — routes and prediction logic
├── generate_models.py              # Script to retrain and save ML models
├── requirements.txt                # Python dependencies
├── Procfile                        # Render deployment config
│
├── models/                         # Trained ML models (pickle files)
│   ├── rf_classifier_categorization.pkl
│   ├── tfidf_vectorizer_categorization.pkl
│   ├── rf_classifier_job_recommendation.pkl
│   └── tfidf_vectorizer_job_recommendation.pkl
│
├── templates/
│   └── resume.html                 # Frontend UI
│
├── UpdatedResumeDataSet.csv.zip    # Training dataset
│
└── Sample Resumes/                 # Test resumes (PDF + TXT)
    ├── Teacher.pdf
    ├── Healthcare.txt
    ├── banking.txt
    └── designer.pdf
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/sayali51/AI-Resume-parser.git
cd AI-Resume-parser

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Unzip dataset and generate models (first time only)
python -c "import zipfile; zipfile.ZipFile('UpdatedResumeDataSet.csv.zip').extractall('extracted_temp')"
move extracted_temp\UpdatedResumeDataSet.csv .
python generate_models.py

# 5. Run the app
python APP.py
```

Open browser at: `http://127.0.0.1:5000`

---

## Sample Output

```
Resume Uploaded: software_engineer.pdf

Category:         Data Science
Recommended Job:  Data Scientist

Extracted Info:
  Name:   John Doe
  Email:  john@example.com
  Phone:  +91 9876543210
  Skills: Python, Machine Learning, SQL, TensorFlow
```

---

## Dataset

- **Source:** UpdatedResumeDataSet (publicly available)
- **Size:** 962 resumes
- **Categories:** 25 job domains including Data Science, HR, Finance, Healthcare, Marketing, and more
- **Preprocessing:** URL removal, special character cleaning, whitespace normalization

---

## Author

**Sayali Kale**
- 📧 sayalikale364@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/sayali-kale-42001a2b1)
- 💻 [GitHub](https://github.com/sayali51)
- 📍 Pune, India

---

## License

This project is open source and available under the [MIT License](LICENSE).
