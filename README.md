# 🔥 DeepRetain AI - Enterprise Customer Churn Predictor

DeepRetain AI is a robust, full-stack Machine Learning application designed to help businesses identify at-risk customers before they leave. By combining a **Random Forest Classifier** with **Google's Gemini 2.5 Flash AI**, the platform provides high-accuracy risk scoring, automated retention strategy generation, and real-time interactive analytics.

---

## 🚀 Core Features

### 1. Predictive Intelligence
* **Batch Prediction:** Upload entire customer datasets (CSV) to get instant churn probability scores for every single row.
* **Universal Matrix:** A dynamic table parser that accepts varied datasets (e.g., Bank records, Subscription data) and maps them to ML features.
* **Probability Scoring:** Provides exact percentage-based risk levels rather than simple binary results.

### 2. AI Strategy Engine (Gemini Powered)
* **Dynamic Report Generation:** Analyzes your current portfolio metrics and uses LLMs to write professional HTML-formatted retention reports.
* **Blue Flame Assistant:** A global AI chat widget accessible on every screen to assist with data interpretation and navigation.

### 3. Sandbox "What-If" Predictor
* **Manual Calculator:** Restored on the main dashboard to allow engineers to manually input customer stats and test the model's reaction in real-time.

### 4. Advanced Data Visualization
* **Real-time Analytics:** Custom-built Gauge, Pie, and Bar charts using **Chart.js** that visualize risk distribution and cohort metric baselines.
* **Speed Optimized:** Heavy mathematical calculations are handled by the Python backend (Pandas) to ensure near-zero UI lag.

### 5. Hybrid Enterprise Database
* **Neon Tech Integration:** Pre-configured to use **SQLite** for local development and **PostgreSQL (Neon)** for production.
* **Persistent Storage:** Saves every portfolio scan and manual calculation to the database for historical tracking.

---

## 🛠️ Tech Stack

### Languages & Frameworks
* **Python 3.10+:** Core backend logic.
* **Flask:** Web framework and API routing.
* **Tailwind CSS:** Modern utility-first frontend styling (CDN-based for performance).
* **JavaScript (ES6):** Client-side interactivity and Chart.js integration.

### Machine Learning & Data Science
* **Scikit-Learn:** Random Forest Classifier for robust, non-linear churn modeling.
* **Pandas & NumPy:** Blazing fast data manipulation and feature engineering.
* **Joblib:** Serialization for model and scaler persistence.

### AI & Infrastructure
* **Google GenAI SDK:** Integration with Gemini 2.5 Flash.
* **SQLAlchemy / Psycopg2:** Dual-engine database connectors.
* **Gunicorn:** Production-grade WSGI server for Render.

---

## 📁 Project Structure

```text
DeepRetain-AI/
├── models/                     # Saved ML Model artifacts
│   ├── churn_model.pkl         # Trained Random Forest Model
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── metrics.json            # Model accuracy and feature metadata
├── customer_churn_dataset.csv  # Data used for training
│      
│          
├── templates/                  # Flask HTML UI Templates
│   ├── index.html              # Main Dashboard (Sandbox + Scans)
│   ├── services.html           # Upload Portal (Active Services)
│   ├── insights.html           # AI Solutions (Gemini Reports)
│   ├── report.html             # Individual Intelligence Report
│   ├── login.html              # Auth View
│   └── register.html           # Auth View
├── .env                        # Private API Keys (DO NOT UPLOAD)
├── .gitignore                  # Prevents secrets from hitting GitHub
├── app.py                      # Main Flask Application & DB Engine
├── train.py                    # ML Model Training Script
|            
├── requirements.txt            # Python Dependencies for Render
├── README.md                   # Documentation
└── deepretain.db               # Local SQLite Database (Auto-generated)