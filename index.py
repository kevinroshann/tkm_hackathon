import torch
import torch.nn as nn
import pandas as pd
import joblib
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch.optim import AdamW
from sklearn.preprocessing import MultiLabelBinarizer

class EmergencyBrain(nn.Module):
    def __init__(self, num_depts):
        super(EmergencyBrain, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        
        # Combined input size: 768 (BERT) + 1 (SOS Flag) = 769
        self.dept_head = nn.Linear(769, num_depts)
        self.sev_head = nn.Linear(769, 1)

    def forward(self, input_ids, attention_mask, sos_flag):
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        
        # Concatenate the SOS flag to the pooled output
        combined = torch.cat((pooled, sos_flag), dim=1)
        
        return torch.sigmoid(self.dept_head(combined)), self.sev_head(combined)

def train_model():
    device = torch.device("cpu")
    df = pd.read_csv("emergency_data.csv")
    
    df['dept_list'] = df['departments'].apply(lambda x: x.split(','))
    mlb = MultiLabelBinarizer()
    y_dept = mlb.fit_transform(df['dept_list'])
    y_sev = df['severity'].values / 100.0
    y_sos = df['sos'].values
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    class EmergencyDataset(Dataset):
        def __init__(self, texts, depts, sevs, sos):
            self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors="pt")
            self.depts = torch.tensor(depts, dtype=torch.float)
            self.sevs = torch.tensor(sevs, dtype=torch.float).unsqueeze(1)
            self.sos = torch.tensor(sos, dtype=torch.float).unsqueeze(1)
        def __len__(self): return len(self.depts)
        def __getitem__(self, i):
            return {
                'ids': self.encodings['input_ids'][i],
                'mask': self.encodings['attention_mask'][i],
                'd': self.depts[i],
                's': self.sevs[i],
                'sos': self.sos[i]
            }

    loader = DataLoader(EmergencyDataset(df['text'], y_dept, y_sev, y_sos), batch_size=16, shuffle=True)
    model = EmergencyBrain(len(mlb.classes_)).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    
    d_criterion = nn.BCELoss()
    s_criterion = nn.MSELoss()

    print(f"Training on {len(df)} samples...")
    for epoch in range(10): # Increased epochs for better variance
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            p_dept, p_sev = model(batch['ids'].to(device), batch['mask'].to(device), batch['sos'].to(device))
            loss = d_criterion(p_dept, batch['d'].to(device)) + s_criterion(p_sev, batch['s'].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "final_model.pt")
    tokenizer.save_pretrained("./final_config")
    joblib.dump(mlb, "final_mlb.joblib")
    print("Training Complete.")

if __name__ == "__main__":
    train_model()