import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- GEMINI CONFIGURATION ---
GENAI_KEY = os.getenv("GEMINI_API_KEY")
if GENAI_KEY:
    genai.configure(api_key=GENAI_KEY)
    # Using user-specified model
    try:
        genai_model = genai.GenerativeModel('models/gemini-2.5-flash-lite-preview-09-2025')
    except:
        genai_model = None
else:
    print("WARNING: GEMINI_API_KEY not found in .env")
    genai_model = None

# Emergency Brain Model
class EmergencyBrain(nn.Module):
    def __init__(self, num_depts):
        super(EmergencyBrain, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.dept_head = nn.Linear(769, num_depts)
        self.sev_head = nn.Linear(769, 1)

    def forward(self, input_ids, attention_mask, sos_flag):
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        combined = torch.cat((pooled, sos_flag), dim=1)
        return torch.sigmoid(self.dept_head(combined)), self.sev_head(combined)

# Load model
device = torch.device("cpu")
mlb = None
tokenizer = None
model = None

def load_model():
    global mlb, tokenizer, model
    try:
        mlb = joblib.load("final_mlb.joblib")
        tokenizer = DistilBertTokenizer.from_pretrained("./final_config")
        model = EmergencyBrain(len(mlb.classes_))
        model.load_state_dict(torch.load("final_model.pt", map_location=device))
        model.eval()
        print("✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("  Please run index.py to train the model first.")
        return False

# Logs storage
LOGS_FILE = "emergency_logs.json"

def save_log(log_entry):
    logs = []
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    
    logs.append(log_entry)
    
    with open(LOGS_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    text = data.get('text', '')
    user_name = data.get('user_name', 'Anonymous')
    user_contact = data.get('user_contact', 'N/A')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Tokenize and predict (initially with sos=0 to get base severity)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    sos_tensor = torch.tensor([[0.0]], dtype=torch.float).to(device)
    
    with torch.no_grad():
        d_probs, s_score = model(
            inputs['input_ids'].to(device),
            inputs['attention_mask'].to(device),
            sos_tensor
        )
        
        d_idx = (d_probs[0] > 0.3).nonzero(as_tuple=True)[0].cpu().numpy()
        depts = mlb.classes_[d_idx].tolist()
        
        # Calculate initial severity score
        score = int(s_score.item() * 100)
        score = max(0, min(100, score))
        
        # Auto-determine SOS based on severity >= 90
        sos_val = 1.0 if score >= 90 else 0.0
        
        # If auto-SOS is triggered, recalculate with SOS flag
        if sos_val == 1.0:
            sos_tensor = torch.tensor([[1.0]], dtype=torch.float).to(device)
            d_probs, s_score = model(
                inputs['input_ids'].to(device),
                inputs['attention_mask'].to(device),
                sos_tensor
            )
            
            d_idx = (d_probs[0] > 0.3).nonzero(as_tuple=True)[0].cpu().numpy()
            depts = mlb.classes_[d_idx].tolist()
            
            score = int(s_score.item() * 100)
            
            # Boost for SOS
            if score < 85:
                score = min(100, score + 15)
            
            score = max(0, min(100, score))
    
    # Determine priority
    if score > 85:
        priority = "CRITICAL"
        action = "IMMEDIATE EMERGENCY DEPLOYMENT"
    elif score > 60:
        priority = "URGENT"
        action = "URGENT RESPONSE REQUIRED"
    else:
        priority = "STANDARD"
        action = "STANDARD DISPATCH"
    
    # Create log entry
    log_entry = {
        'id': datetime.now().strftime('%Y%m%d%H%M%S%f'),
        'timestamp': datetime.now().isoformat(),
        'user_name': user_name,
        'user_contact': user_contact,
        'complaint': text,
        'sos': bool(sos_val),
        'departments': depts if len(depts) > 0 else ['REVIEW_NEEDED'],
        'initial_depts': depts if len(depts) > 0 else ['REVIEW_NEEDED'], # Track original
        'backup_requests': [], # Track inter-dept requests
        'severity': score,
        'priority': priority,
        'action': action,
        'resolved': False,
        'resolved_by': [],
        'resolved_at': None
    }
    
    save_log(log_entry)
    
    return jsonify(log_entry)

@app.route('/logs', methods=['GET'])
def get_logs():
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, 'r') as f:
            try:
                logs = json.load(f)
                # Return in reverse order (newest first)
                return jsonify(logs[::-1])
            except:
                return jsonify([])
    return jsonify([])

@app.route('/logs/<department>', methods=['GET'])
def get_department_logs(department):
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, 'r') as f:
            try:
                logs = json.load(f)
                # Filter logs for specific department
                dept_logs = [log for log in logs if department.upper() in [d.upper() for d in log.get('departments', [])]]
                return jsonify(dept_logs[::-1])
            except:
                return jsonify([])
    return jsonify([])

@app.route('/resolve/<case_id>', methods=['POST'])
def resolve_case(case_id):
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except:
                return jsonify({'error': 'Failed to read logs'}), 500
        
        # Get Department from request
        data = request.json or {}
        resolving_dept = data.get('department') # e.g. "Fire"
        
        # Find and update the case
        updated = False
        case_fully_resolved = False
        
        for log in logs:
            if log.get('id') == case_id:
                # Ensure resolved_by list exists (backward compatibility)
                if 'resolved_by' not in log:
                    log['resolved_by'] = []
                
                # Add dept to resolved list if provided and not already there
                if resolving_dept:
                    if resolving_dept not in log['resolved_by']:
                        log['resolved_by'].append(resolving_dept)
                else:
                    # Legacy fallback: if no dept provided, resolve all (or just mark resolved)
                    # Ideally we force dept providing, but for safety:
                    pass

                # Check if ALL assigned departments have resolved
                assigned_depts = log.get('departments', [])
                # clean up department names for comparison (case insensitive often safer but sticking to exact matches for now)
                
                # Check if set(resolved_by) covers set(assigned_depts)
                is_all_resolved = all(d in log['resolved_by'] for d in assigned_depts)
                
                if is_all_resolved:
                    log['resolved'] = True
                    log['resolved_at'] = datetime.now().isoformat()
                    case_fully_resolved = True
                
                updated = True
                break
        
        if updated:
            with open(LOGS_FILE, 'w') as f:
                json.dump(logs, f, indent=2)
            return jsonify({'success': True, 'id': case_id, 'fully_resolved': case_fully_resolved})
        else:
            return jsonify({'error': 'Case not found'}), 404
    
    return jsonify({'error': 'No logs file'}), 404

@app.route('/generate_advice', methods=['POST'])
def generate_advice():
    if not genai_model:
        return jsonify({"error": "AI Model not configured (API Key missing or Invalid Model)"}), 503
    
    data = request.json
    complaint = data.get('complaint', '')
    dept = data.get('departments', [])
    severity = data.get('severity', 0)
    audience = data.get('audience', 'responder') # 'civilian' or 'responder'
    
    if audience == 'civilian':
        prompt = f"""
        Act as a Calm Emergency Dispatcher.
        Context:
        - Incident: {complaint}
        - Severity: {severity}/100
        
        Task:
        Provide 3-4 simple, immediate safety steps for a CIVILIAN (potentially a child) involved in this situation.
        Focus on "Stay Calm", "Safety First", and "Help is coming".
        Use very simple language. Short sentences. No complications.
        """
    else:
        # Responder / Tactical
        prompt = f"""
        Act as an Emergency Response Tactical Commander.
        Context:
        - Incident: {complaint}
        - Assigned Depts: {', '.join(dept)}
        - Severity: {severity}/100
        
        Task:
        Generate a precise, immediate tactical checklist for the response team.
        Format as a markdown list. Keep it under 6 brief points.
        Include specific safety instructions and equipment requirements.
        Do not include greetings or filler text.
        """
    
    try:
        response = genai_model.generate_content(prompt)
        return jsonify({"advice": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/request_backup', methods=['POST'])
def request_backup():
    data = request.json
    origin = data.get('origin_dept', 'Unknown')
    target = data.get('target_dept', 'General')
    message = data.get('message', 'Assistance required')
    ref_id = data.get('ref_id', 'N/A')
    
    # MERGE LOGIC: Update existing case instead of creating new one
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, 'r') as f:
                logs = json.load(f)
            
            updated = False
            for log in logs:
                if log.get('id') == ref_id:
                    # 1. Add target to assigned departments if not present
                    current_depts = log.get('departments', [])
                    if target not in current_depts:
                        current_depts.append(target)
                        log['departments'] = current_depts
                    
                    # 2. Add alert to backup_requests
                    if 'backup_requests' not in log:
                        log['backup_requests'] = []
                    
                    log['backup_requests'].append({
                        "from": origin,
                        "to": target,
                        "message": message,
                        "timestamp": datetime.now().strftime('%H:%M:%S')
                    })
                    updated = True
                    break
            
            if updated:
                with open(LOGS_FILE, 'w') as f:
                    json.dump(logs, f, indent=2)
                return jsonify({"success": True, "merged": True})
            else:
                return jsonify({"error": "Original case not found"}), 404
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No logs"}), 404

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("=" * 50)
    print("EMERGENCY RESPONSE AI - API SERVER")
    print("=" * 50)
    
    if load_model():
        print("\n🚀 Starting API server on http://localhost:5000")
        print("=" * 50)
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("\n⚠️  Cannot start server without model")
        print("   Run 'python index.py' first to train the model")