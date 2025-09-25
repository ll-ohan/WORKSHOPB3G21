#!/bin/bash
export DOCKER_SERVICE_URL="http://localhost:8080"
cd "$(dirname "$0")"
python main.py