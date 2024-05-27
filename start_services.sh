#!/bin/bash
conda init
conda activate dissertation_rag

# Start FastAPI service
cd dashboard/fast-api
uvicorn main_api:app --host 0.0.0.0 --port 8082 &
fastapi_pid=$!

# Start Flask dashboard
cd ..
cd flask-app
python app.py &
flask_pid=$!

# Wait for both services to stop
wait

# Function to stop services
function stop_services {
    kill $fastapi_pid
    kill $flask_pid
}

# Call the function when this script is interrupted
trap stop_services INT