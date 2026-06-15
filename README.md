# 🔍 AI Fairness Auditor Pro

> Detect, explain, and fix bias in your AI before it harms real people.

---

## 🚨 The Problem

When companies build AI for hiring, loans, insurance, or any decision-making — they train it on historical data. That data often reflects decades of human bias. The AI learns those patterns and starts discriminating against women, minorities, or older people — without anyone realizing it.

Big companies like Google and Microsoft have entire ethics teams to prevent this. **Small startups have nobody. We built that nobody's team.**

---

## ✅ What It Does

A complete end-to-end fairness auditing platform that:
- Detects bias **before** training (dataset audit)
- Detects bias **after** training (model audit)
- Tests **live deployed models** without seeing their code
- **Fixes** the biased dataset automatically
- **Explains** everything in plain English
- Generates a professional **audit report** with final verdict

---

## 🎯 Key Features

### 📊 Pre-Training Dataset Audit
Upload your dataset in any format — CSV, Excel, JSON, XML, HTML, PDF, or plain text. Automatically detects protected attributes (gender, age, race, religion, disability) and computes fairness metrics with interactive charts.

### 🛠 Fix Dataset
Diagnoses the type of bias, generates a custom repair algorithm using AI, applies outcome-aware stratified oversampling, validates the fix actually worked, and provides a full before vs after graphical comparison. Download the fixed dataset as CSV.

### 🎯 Post-Training Model Audit
Upload your model's predictions file. Audits model outputs for fairness rather than raw data. Also computes accuracy, precision, recall and F1 score.

### 🌐 Black-Box API Stress Test
Give the URL of your live deployed model. Sends identical requests multiple times changing only the protected attribute — if responses differ, bias is caught in real time. No model code access needed.

### 🔬 Sandbox Counterfactual Testing
Paste your model's prediction function as Python code. Runs it against synthetic profiles where only the protected attribute changes. Measures counterfactual bias with full AI explanation.

### 🎛 What-If Bias Simulator
Drag sliders to simulate dataset rebalancing. Instantly see projected bias score improvement without touching real data.

### ⚖️ Appeal Engine
Individuals who received an AI-driven rejection can upload their document and the company's policy. AI compares both and returns a fit score, matched requirements, rejection reasons, fairness concerns, and appeal recommendation.

### 📄 AI Audit Reports
9-section structured report with final verdict — PASS, INCONCLUSIVE, FAIL, or CRITICAL. Exportable as TXT or PDF.

---

## 📐 Core Fairness Metrics

| Metric | Formula | Threshold |
|--------|---------|-----------|
| Statistical Parity Difference (SPD) | max_rate - min_rate | > 0.10 = bias |
| Disparate Impact (DI) | min_rate / max_rate | < 0.80 = fails 80% rule |
| Bias Score | Custom piecewise formula | 0-15 LOW, 15-35 MEDIUM, 35-60 HIGH, 60+ CRITICAL |

SPD and DI are based on the **US EEOC 80% Rule** — the same legal standard used in employment discrimination cases.

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| Frontend | HTML5, CSS3, JavaScript, Chart.js 4.4 |
| Database | MongoDB (PyMongo) |
| AI — Primary | Groq API (Llama 3.3 70B) |
| AI — Fallback | Google Gemini 2.0 Flash |
| PDF Export | ReportLab |
| File Parsing | pandas, pdfplumber, xml.etree, regex |
| Security | Sandboxed subprocess, session auth |

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/AI-FAIRNESS-AUDITOR-PRO.git
cd AI-FAIRNESS-AUDITOR-PRO
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root folder:
```env
MONGO_URI=your_mongodb_connection_string
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
SECRET_KEY=your_secret_key
```

> ℹ️ The tool uses **Groq as primary AI** and **Gemini as fallback**. You need at least one API key to run AI features. The tool works without AI but explanations will use fallback text.

### 5. Run Application
```bash
python app.py
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📁 Project Structure
AI-FAIRNESS-AUDITOR-PRO/

│

├── app.py                  # Main Flask application

├── requirements.txt        # Python dependencies

├── .env                    # Environment variables (not committed)

│

├── templates/

│   ├── index.html          # Landing page

│   ├── login.html          # Login page

│   ├── register.html       # Register page

│   └── dashboard.html      # Main dashboard

│

├── static/

│   ├── style.css           # Main stylesheet

│   ├── style_fix.css       # Fix dataset styles

│   ├── dashboard.js        # Main JavaScript

│   └── fix_dataset.js      # Fix dataset module

│

└── uploads/                # Temporary file uploads

---

## 🔄 How It Works
User uploads dataset

↓

Python detects protected attributes automatically

↓

Computes SPD, DI, group distributions, outcome rates

↓

Chart.js renders interactive graphs in browser

↓

Groq / Gemini generates plain English explanation per chart

↓

Full audit report generated with final verdict

↓

User can fix dataset, simulate changes, or export PDF

---

## 🧪 Supported File Formats

| Format | Method |
|--------|--------|
| CSV | pandas direct read |
| Excel (XLSX/XLS) | pandas direct read |
| JSON | json + pandas normalize |
| XML | xml.etree parser |
| HTML | pandas read_html |
| PDF | pdfplumber text extraction |
| TXT | Smart 5-strategy regex parser |

---

## 🎯 Use Cases

- Startups auditing hiring AI before deployment
- Fintech companies checking loan approval models
- Healthcare AI fairness validation
- EU AI Act compliance documentation
- Academic research in responsible AI
- Individual appeal against AI-driven rejection

---

## 🔮 Roadmap

- [ ] Digital fairness certificate with public verification URL
- [ ] Proxy variable detection (bias hiding in non-protected columns)
- [ ] Intersectional bias detection (combinations of attributes)
- [ ] Multilingual bias auditing (Hindi, Tamil, Arabic)
- [ ] Continuous monitoring API for production models
- [ ] Bias fingerprinting database

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Built By

**Parth Koli**
Final Year Computer Science Student — APSIT, Thane
