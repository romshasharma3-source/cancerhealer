#!/bin/bash
cd /home/site/wwwroot
pip install --upgrade pip
pip install -r requirements.txt
exec python -m streamlit run app_2.py --server.port=8000 --server.address=0.0.0.0