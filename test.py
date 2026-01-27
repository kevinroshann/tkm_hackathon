import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import joblib

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

def run_test():
    device = torch.device("cpu")
    print("Loading AI...")
    
    try:
        mlb = joblib.load("final_mlb.joblib")
        tokenizer = DistilBertTokenizer.from_pretrained("./final_config")
        model = EmergencyBrain(len(mlb.classes_))
        model.load_state_dict(torch.load("final_model.pt", map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}. Please run train.py first.")
        return
    
    print("Ready. Type your emergency description below.")
    while True:
        text = input("\nComplaint (or 'q' to exit): ")
        if text.lower() in ['q', 'quit', 'exit']: break
        
        sos_input = input("Is this an SOS? (1 for Yes, 0 for No): ")
        sos_val = 1.0 if sos_input == "1" else 0.0
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        sos_tensor = torch.tensor([[sos_val]], dtype=torch.float).to(device)
        
        with torch.no_grad():
            d_probs, s_score = model(inputs['input_ids'].to(device), inputs['mask'].to(device) if 'mask' in inputs else inputs['attention_mask'].to(device), sos_tensor)
            
            d_idx = (d_probs[0] > 0.3).nonzero(as_tuple=True)[0].cpu().numpy()
            depts = mlb.classes_[d_idx]
            
            # Final score calculation
            score = int(s_score.item() * 100)
            
            # Manual boost if SOS was selected but model was conservative
            if sos_val == 1 and score < 85:
                score = min(100, score + 15)
            
            score = max(0, min(100, score))
            
        print("-" * 40)
        print(f"ANALYSIS RESULTS [SOS: {'YES' if sos_val == 1 else 'NO'}]")
        print(f"Departments: {', '.join(depts) if len(depts) > 0 else 'Dispatcher Review Needed'}")
        print(f"Severity Score: {score}/100")
        
        if score > 85: print("ACTION: IMMEDIATE EMERGENCY DEPLOYMENT")
        elif score > 60: print("ACTION: URGENT RESPONSE REQUIRED")
        else: print("ACTION: STANDARD DISPATCH")
        print("-" * 40)

if __name__ == "__main__":
    run_test()