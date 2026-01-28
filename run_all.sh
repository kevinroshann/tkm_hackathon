#!/bin/bash

python3 datset_creator.py
python3 index.py
python3 train_model.py
python3 patient_model.py

# Run backend API server in background
python3 api_server.py &

# Run streamlit apps in background
streamlit run public_app.py &
streamlit run admin/app.py &

