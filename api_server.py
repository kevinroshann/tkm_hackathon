import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import os

app = Flask(__name__)
CORS(app)

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
        'severity': score,
        'priority': priority,
        'action': action,
        'resolved': False,
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
        
        # Find and update the case
        updated = False
        for log in logs:
            if log.get('id') == case_id:
                log['resolved'] = True
                log['resolved_at'] = datetime.now().isoformat()
                updated = True
                break
        
        if updated:
            with open(LOGS_FILE, 'w') as f:
                json.dump(logs, f, indent=2)
            return jsonify({'success': True, 'id': case_id})
        else:
            return jsonify({'error': 'Case not found'}), 404
    
    return jsonify({'error': 'No logs file'}), 404

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