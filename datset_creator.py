import pandas as pd
import random

def create_csv():
    data = []
    scenarios = [
        ("Active shooter in the shopping mall", ["Police", "Medical"], 98),
        ("Large warehouse fire with chemical containers", ["Fire", "Environmental", "Medical"], 95),
        ("Stabbing incident during a street fight", ["Police", "Medical"], 88),
        ("Cyber attack on city water treatment plant", ["Cyber", "Utilities", "Public Works"], 90),
        ("Oil tanker leaking in the harbor", ["Coastal Guard", "Environmental"], 82),
        ("Child found alone and shivering in the park", ["Social Services", "Police"], 25),
        ("Power lines down on a flooded street", ["Utilities", "Fire", "Public Works"], 75),
        ("Aggressive pitbull attacking a pedestrian", ["Animal Control", "Medical", "Police"], 65),
        ("Elderly person collapsed with chest pain", ["Medical"], 85),
        ("Car accident, vehicle flipped and smoking", ["Fire", "Medical", "Police"], 80),
        ("Illegal dumping of toxic waste in the river", ["Environmental", "Police"], 70),
        ("Suicidal individual on the bridge ledge", ["Police", "Medical", "Social Services"], 90),
        ("Boat sinking with 5 passengers on board", ["Coastal Guard", "Medical"], 95),
        ("Pothole reported on a quiet residential street", ["Public Works"], 8),
        ("Someone trying to hack into my bank account", ["Cyber", "Police"], 45),
        ("Domestic dispute with loud screaming", ["Police", "Social Services"], 55),
        ("Street light out at a busy intersection", ["Public Works", "Utilities"], 12),
        ("Homeless camp needing a welfare check", ["Social Services"], 20),
        ("Raccoon trapped in a residential chimney", ["Animal Control"], 15),
        ("Explosion heard in the industrial district", ["Fire", "Police", "Medical", "Utilities"], 99),
        ("Gas smell in the apartment basement", ["Fire", "Utilities"], 75),
        ("Identity theft and fraudulent credit cards", ["Cyber", "Police"], 40),
        ("Broken water main flooding the downtown street", ["Utilities", "Public Works"], 60),
        ("Vandalism and graffiti on the school wall", ["Police", "Public Works"], 10),
        ("Wildfire approaching residential area", ["Fire", "Police", "Environmental"], 92),
        ("Testing connection", ["Utilities"], 5)
    ]

    locations = ["Main St", "Downtown", "the East Side", "the Suburbs", "the Port", "Oak Road", "the Mall"]
    
    for _ in range(800):
        base_text, depts, base_sev = random.choice(scenarios)
        loc = random.choice(locations)
        sos_flag = random.choice([0, 0, 0, 1]) # 25% chance of SOS
        
        prefix = random.choice(["Reporting a", "There is a", "URGENT:", "I see a", "Calling about a"])
        text = f"{prefix} {base_text.lower()} near {loc}."
        
        # If SOS is 1, boost severity significantly
        if sos_flag == 1:
            final_sev = min(100, base_sev + 20)
        else:
            final_sev = max(1, min(100, base_sev + random.randint(-8, 8)))
        
        data.append([text, ",".join(depts), final_sev, sos_flag])

    df = pd.DataFrame(data, columns=["text", "departments", "severity", "sos"])
    df.to_csv("emergency_data.csv", index=False)
    print("Dataset 'emergency_data.csv' created with 800 entries.")

if __name__ == "__main__":
    create_csv()