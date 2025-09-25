#!/bin/bash
echo "Démarrage BatMap..."

# API en arrière-plan
cd api && python main.py &
API_PID=$!

sleep 3

# Streamlit
cd .. && streamlit run app.py --server.port 8501 &
STREAMLIT_PID=$!

echo "Système démarré:"
echo "- API: http://localhost:8000"
echo "- Interface: http://localhost:8501"

trap "kill $API_PID $STREAMLIT_PID" EXIT
wait