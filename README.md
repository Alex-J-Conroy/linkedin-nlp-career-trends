# 🧠 LinkedIn NLP Career Trends — Skill Clustering & Workforce Benchmarking

## 📌 Overview

This project analyses **public LinkedIn profiles** from professionals at a major consulting firm to derive **skill trends, career levels, and team structure insights** using **NLP and unsupervised learning**. It was created to inform **performance evaluations, pay equity conversations, and skill alignment efforts** across technical roles.

> **Key Question:** How can public LinkedIn data be transformed into structured, explainable insights about workforce skills and progression?

---

## 👤 Author

**Alex Conroy**  
📍 Individual Project (Internal Application)  
📊 Data Science | Workforce Analytics | NLP

---

## 🧾 Data Summary

- **Source:** Public LinkedIn profiles
- **Scope:** 200+ professionals in KPMG Australia's Tech Group
- **Fields Extracted:**
  - Role titles
  - Skills and endorsements
  - Role timelines
  - Education histories

📄 Large-scale profiling via [YData Profiling](https://github.com/ydataai/ydata-profiling) also performed but not included due to file size.

---

## ⚙️ Methodology

### 🧹 Preprocessing
- Cleaned role titles, normalised capitalisation
- Parsed job and education timelines
- Extracted and tokenised skill keywords

### 🧠 NLP + Clustering
- TF-IDF vectorisation on job descriptions
- KMeans clustering of profiles into skill families
- Summarised each cluster by dominant terms and sample roles

### 📈 Timeline & Level Analysis
- Inferred **career stage** based on:
  - Time since graduation
  - Time in current and prior roles
- Compared **title vs. seniority delta** across clusters

---

## 💡 Key Outcomes

✅ Identified **clear skill clusters** across roles  
✅ Inferred **career levels** and **time-in-role estimates**  
✅ Highlighted inconsistencies in role naming vs. experience  
✅ Supported a **data-driven approach to levelling and pay review**

---

## 🗂 Project Structure
linkedin-nlp-career-trends/ ├── data/ │ ├── result.csv # Cleaned LinkedIn profiles with inferred timelines │ └── tech_group.csv # Source profile metadata │ ├── notebooks/ │ └── main_analysis.ipynb # NLP, clustering, and time-in-role analysis │ ├── src/ │ └── modeling.py # TF-IDF, clustering, and summarisation logic │ ├── utils/ │ ├── preprocessing.py # Text cleaning, timeline parsing │ └── visualisation.py # Wordclouds, heatmaps, cluster visuals │ ├── report/ │ └── Role_profile_report.html # Interactive EDA report (YData Profiling) │ ├── requirements.txt └── README.md

 🔮 Future Work
- Improve entity extraction for role titles and skills using transformer-based models (e.g., BERT-NER) instead of basic tokenisation.
- Include more contextual signals such as certifications, project keywords, or team/manager tagging (if available).
- Handle non-linear career paths more accurately (e.g., job switches, career breaks).
- Extend clustering to incorporate graph-based relationships between roles or skill sets.
- Develop a live dashboard for internal use with filtering by business unit, tenure, or location.

⚠️ Limitations
- Career level estimates are inferred from public data (graduation dates, job timelines), which may be inaccurate or incomplete.
- Job title standardisation is approximate — inconsistent LinkedIn formatting introduces noise.
- Skills and endorsements on LinkedIn are often user-curated and not validated, leading to potential bias or inflation.
- Small sample size (limited to publicly available profiles from a single business unit) may not generalise across the broader workforce.
- Clustering is based purely on textual features (TF-IDF) — lacks semantic richness from embeddings or business context.

📄 License
This project uses publicly accessible data only.
No confidential or proprietary information is included.
