import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import json
import uuid
import os
import re
import math
from dotenv import load_dotenv
from google import genai

# --- INITIALIZE ENV & GEMINI ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL") 

gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini API connected successfully!")
    except Exception as e:
        print(f"⚠️ Gemini init failed: {e}")

app = Flask(__name__)
# CRUCIAL FOR LOGIN: Secret key allows Flask to encrypt the user's session securely
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "deepretain-fallback-secret-key-123")

# --- HYBRID DATABASE ENGINE ---
def execute_db(query, params=(), fetchone=False, fetchall=False, commit=False):
    is_postgres = False
    if DATABASE_URL:
        try:
            import psycopg2
            is_postgres = True
        except ImportError:
            pass

    if is_postgres:
        import psycopg2
        from psycopg2.extras import DictCursor
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=DictCursor)
        formatted_query = query.replace('?', '%s')
    else:
        import sqlite3
        conn = sqlite3.connect('deepretain.db', check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        formatted_query = query

    try:
        cursor.execute(formatted_query, params)
        result = None
        if fetchone: 
            row = cursor.fetchone()
            result = dict(row) if row else None
        if fetchall: 
            result = [dict(row) for row in cursor.fetchall()]
        if commit: 
            conn.commit()
    except Exception as e:
        print(f"Database Execution Error: {e}")
        result = [] if fetchall else None
    finally:
        conn.close()
        
    return result

def init_db():
    execute_db('''CREATE TABLE IF NOT EXISTS services (id TEXT PRIMARY KEY, service_name TEXT, service_type TEXT, status TEXT, churn_prob REAL, display_features TEXT)''', commit=True)
    execute_db('''CREATE TABLE IF NOT EXISTS reports (service_id TEXT PRIMARY KEY, predictions_json TEXT, stats_json TEXT)''', commit=True)
    execute_db('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT, role TEXT)''', commit=True)
    
    if DATABASE_URL:
        try:
            execute_db('''CREATE TABLE IF NOT EXISTS recent_calcs (id SERIAL PRIMARY KEY, inputs_json TEXT, result REAL)''', commit=True)
        except: pass
    else:
        execute_db('''CREATE TABLE IF NOT EXISTS recent_calcs (id INTEGER PRIMARY KEY AUTOINCREMENT, inputs_json TEXT, result REAL)''', commit=True)

    # Automatically create the default admin account if it doesn't exist
    admin_exists = execute_db("SELECT * FROM users WHERE username = 'admin'", fetchone=True)
    if not admin_exists:
        default_hash = generate_password_hash('admin@newTRUE1')
        execute_db("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", ('admin', default_hash, 'Core Admin'), commit=True)

init_db()

# --- HELPER FUNCTIONS ---
def clean_col_str(c): return re.sub(r'[^a-zA-Z0-9]', '', str(c)).lower()

def safe_float(val):
    try:
        if pd.isna(val) or val is None: return 0.0
        val_str = str(val).replace(',', '').replace('$', '').strip()
        if val_str == '': return 0.0
        f = float(val_str)
        if math.isnan(f): return 0.0
        return f
    except Exception: return 0.0

print("Loading DeepRetain AI Model...")
try:
    model = joblib.load('models/churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/metrics.json', 'r') as f: model_metrics = json.load(f)
    model_accuracy = model_metrics.get("model_accuracy", 0)
    total_customers = model_metrics.get("total_customers", 0)
    features = model_metrics.get("features", ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction'])
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ WARNING: Model files not found! Please run 'python train.py' first.")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    model, scaler = RandomForestClassifier(), StandardScaler()
    model_accuracy, total_customers, features = 0, 0, ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']

# --- AUTHENTICATION GATEKEEPER ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- ROUTES ---
@app.route('/')
@login_required
def dashboard():
    try:
        services = execute_db('SELECT * FROM services', fetchall=True)
        if not services: services = []
        services = services[::-1] 
    except:
        services = []
        
    metrics = {"model_accuracy": model_accuracy, "total_customers": total_customers}
    return render_template('index.html', username=session.get('username', 'Admin'), role=session.get('role', 'AI Engineer'), active_services=services, metrics=metrics, features=features)

@app.route('/services', methods=['GET', 'POST'])
@login_required
def manage_services():
    try:
        services = execute_db('SELECT * FROM services', fetchall=True)
        if not services: services = []
        services = services[::-1]
    except:
        services = []

    if request.method == 'GET':
        return render_template('services.html', username=session.get('username', 'Admin'), role=session.get('role', 'AI Engineer'), services=services)
    
    if request.method == 'POST':
        service_name = request.form.get('service_name', 'Unnamed Service')
        service_type = request.form.get('service_type', 'Standard')
        predictions_list = []
        
        if 'dataset' in request.files and request.files['dataset'].filename != '':
            file = request.files['dataset']
            df_new = pd.read_csv(file, encoding='utf-8-sig')
            df_new.columns = [str(c).replace('\ufeff', '').strip() for c in df_new.columns]
            
            client_id_col = None
            for c in df_new.columns:
                if clean_col_str(c) in ['customerid', 'clientid', 'id', 'userid']:
                    client_id_col = c; break
            
            display_features = [c for c in df_new.columns if c != client_id_col]
            if not display_features: display_features = df_new.columns.tolist()

            col_mapping = {}
            matched_count = 0
            for f in features:
                clean_f = clean_col_str(f)
                matched_col = None
                for uc in df_new.columns:
                    if clean_f in clean_col_str(uc) or clean_col_str(uc) in clean_f:
                        matched_col = uc; matched_count += 1; break
                col_mapping[f] = matched_col

            if matched_count == 0:
                return f"""<div style="font-family:sans-serif; padding:40px; text-align:center;"><h2 style="color:#d93025;">Dataset Mismatch Error</h2><p>Your AI model was trained to predict using columns like <b>{features[0]}</b>.</p><p>You uploaded a file containing different columns like <b>{df_new.columns[0]}</b>.</p><br><a href="/services" style="color:white; background:#1a73e8; padding:10px 20px; border-radius:6px; text-decoration:none; font-weight:bold;">Go Back</a></div>""", 400

            parsed_display_data, ml_data = [], []
            for index, row in df_new.iterrows():
                row_ml = {f: safe_float(row[col_mapping[f]]) if col_mapping[f] else 0.0 for f in features}
                ml_data.append(row_ml)
                parsed_display_data.append({c: str(row[c]) for c in display_features})
                
            X_new = pd.DataFrame(ml_data)[features]
            X_new_scaled = scaler.transform(X_new)
            probs = model.predict_proba(X_new_scaled)[:, 1] * 100
            
            for i, row_display in enumerate(parsed_display_data):
                churn_prob = float(probs[i])
                status = "Critical" if churn_prob > 70 else ("Moderate" if churn_prob > 40 else "Safe")
                client_id = df_new.at[i, client_id_col] if client_id_col else f"Client_{i+1}"
                predictions_list.append({"client_identifier": str(client_id), "display_data": row_display, "ml_data": ml_data[i], "status": status, "churn_prob": round(churn_prob, 2)})
                
            df_preds = pd.DataFrame(predictions_list)
            f1_name = col_mapping[features[0]] if col_mapping.get(features[0]) else display_features[0]
            f2_name = col_mapping[features[4]] if len(features) > 4 and col_mapping.get(features[4]) else (display_features[1] if len(display_features) > 1 else display_features[0])
            
            stats = {
                "c_count": int((df_preds['status'] == 'Critical').sum()),
                "m_count": int((df_preds['status'] == 'Moderate').sum()),
                "s_count": int((df_preds['status'] == 'Safe').sum()),
                "avg_f1": round(float(X_new[features[0]].mean()), 2),
                "avg_f2": round(float(X_new[features[4]].mean()), 2) if len(features) > 4 else 0.0,
                "f1_name": f1_name, "f2_name": f2_name
            }
        else:
            manual_data = {f: safe_float(request.form.get(f)) for f in features}
            X_new = pd.DataFrame([manual_data])[features]
            X_new_scaled = scaler.transform(X_new)
            churn_prob = float(model.predict_proba(X_new_scaled)[0][1] * 100)
            status = "Critical" if churn_prob > 70 else ("Moderate" if churn_prob > 40 else "Safe")
            predictions_list.append({"client_identifier": "Manual_Entry_01", "display_data": manual_data, "ml_data": manual_data, "status": status, "churn_prob": round(churn_prob, 2)})
            display_features = features
            stats = {"c_count": 1 if status=='Critical' else 0, "m_count": 1 if status=='Moderate' else 0, "s_count": 1 if status=='Safe' else 0, "avg_f1": manual_data[features[0]], "avg_f2": manual_data[features[4]] if len(features)>4 else 0, "f1_name": features[0], "f2_name": features[4] if len(features)>4 else 'Feature 2'}

        overall_prob = sum(p['churn_prob'] for p in predictions_list) / len(predictions_list) if predictions_list else 0
        overall_status = "Critical" if overall_prob > 70 else ("Moderate" if overall_prob > 40 else "Safe")
        report_id = str(uuid.uuid4())
        
        execute_db('INSERT INTO services (id, service_name, service_type, status, churn_prob, display_features) VALUES (?, ?, ?, ?, ?, ?)', 
                   (report_id, service_name, service_type, overall_status, round(overall_prob, 2), json.dumps(display_features)), commit=True)
        execute_db('INSERT INTO reports (service_id, predictions_json, stats_json) VALUES (?, ?, ?)', 
                   (report_id, json.dumps(predictions_list), json.dumps(stats)), commit=True)
        
        return redirect(url_for('view_report', report_id=report_id))

@app.route('/report/service/<report_id>')
@login_required
def view_report(report_id):
    service_row = execute_db('SELECT * FROM services WHERE id = ?', (report_id,), fetchone=True)
    report_row = execute_db('SELECT * FROM reports WHERE service_id = ?', (report_id,), fetchone=True)
    
    if not service_row or not report_row: return "Report not found", 404
    
    service_data = dict(service_row)
    service_data['display_features'] = json.loads(service_data['display_features'])
    
    predictions = json.loads(report_row['predictions_json'])
    if isinstance(predictions, str): predictions = json.loads(predictions)
    
    stats = json.loads(report_row['stats_json'])
    if isinstance(stats, str): stats = json.loads(stats)
    
    valid_preds = [p.get('display_data', {}) for p in predictions if isinstance(p, dict)]
    df_train_mock = pd.DataFrame(valid_preds).apply(pd.to_numeric, errors='coerce')
    medians = df_train_mock.median().to_dict() if not df_train_mock.empty else {}

    return render_template('report.html', username=session.get('username', 'Admin'), role=session.get('role', 'AI Engineer'), report_data=service_data, risky_clients=predictions, medians=medians, stats=stats, features=features)

@app.route('/insights')
@login_required
def insights():
    if not gemini_client:
        ai_report = "<h2>API Key Missing</h2><p>Please add your GEMINI_API_KEY to the .env file.</p>"
    else:
        try:
            services = execute_db('SELECT * FROM services', fetchall=True) or []
            prompt = f"Write a professional HTML report (using strictly <h3>, <p>, <ul>, <li> tags ONLY. Do NOT use markdown like ** or *). Analyzing customer retention. Our ML accuracy is {model_accuracy}% and we have {len(services)} services. Give 3 actionable retention tips."
            response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            ai_report = response.text.replace("```html", "").replace("```", "").strip()
        except Exception as e:
            ai_report = f"<h2>API Connection Error</h2><p>{str(e)}</p>"
    return render_template('insights.html', username=session.get('username', 'Admin'), role=session.get('role', 'AI Engineer'), ai_report=ai_report)

@app.route('/api/manual_predict', methods=['POST'])
@login_required
def api_manual_predict():
    try:
        data = {k: safe_float(v) for k, v in request.json.items()}
        X_new = pd.DataFrame([data])
        for col in features:
            if col not in X_new.columns: X_new[col] = 0.0
        X_new = X_new[features]
        X_new_scaled = scaler.transform(X_new)
        churn_prob = round(float(model.predict_proba(X_new_scaled)[0][1] * 100), 1)
        
        execute_db('INSERT INTO recent_calcs (inputs_json, result) VALUES (?, ?)', (json.dumps(data), churn_prob), commit=True)
        return jsonify({"success": True, "result": churn_prob})
    except Exception as e: return jsonify({"error": str(e)}), 400

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    msg = request.json.get('message', '')
    if not gemini_client: return jsonify({"reply": "API Key missing."})
    try:
        response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=f"You are Blue Flame, a retention AI. User says: '{msg}'. Reply concisely.")
        return jsonify({"reply": response.text})
    except Exception as e: return jsonify({"reply": f"Error: {str(e)}"})


# --- AUTHENTICATION ROUTES ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Verify user against the database
        user = execute_db("SELECT * FROM users WHERE username = ?", (username,), fetchone=True)
        
        if user and check_password_hash(user['password_hash'], password):
            session['logged_in'] = True
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid username or password. Please try again."
            
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register(): 
    error = None
    success = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role', 'Data Analyst') # Default role if they don't pick one
        
        if not username or not password:
            error = "Username and password are required."
        else:
            # Check if username is taken
            existing_user = execute_db("SELECT * FROM users WHERE username = ?", (username,), fetchone=True)
            if existing_user:
                error = "Username already exists. Please choose a different one."
            else:
                # Hash the password and save
                hashed_pw = generate_password_hash(password)
                execute_db("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                           (username, hashed_pw, role), commit=True)
                success = "Registration successful! You can now log in."
                
    return render_template('register.html', error=error, success=success)

@app.route('/logout')
def logout(): 
    session.clear() # This destroys the login session
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)