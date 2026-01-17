#!/bin/bash

# Start Music Cluster development environment
# Starts both the FastAPI backend and Svelte frontend

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Music Cluster development environment...${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please create it first:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if node_modules exists in ui directory
if [ ! -d "ui/node_modules" ]; then
    echo "Installing Node.js dependencies..."
    cd ui
    npm install
    cd ..
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${BLUE}Shutting down...${NC}"
    kill $API_PID $UI_PID 2>/dev/null || true
    exit
}

# Trap Ctrl+C
trap cleanup INT TERM

# Start FastAPI server
echo -e "${GREEN}Starting FastAPI server on http://localhost:8000${NC}"
./venv/bin/uvicorn music_cluster.api:app --reload --port 8000 &
API_PID=$!

# Wait a moment for API to start
sleep 2

# Start Svelte dev server
echo -e "${GREEN}Starting Svelte dev server on http://localhost:1420${NC}"
cd ui
npm run dev &
UI_PID=$!
cd ..

echo -e "${GREEN}✓ Both servers are running!${NC}"
echo -e "${BLUE}API: http://localhost:8000${NC}"
echo -e "${BLUE}UI: http://localhost:1420${NC}"
echo -e "${BLUE}API Docs: http://localhost:8000/docs${NC}"
echo -e "\nPress Ctrl+C to stop both servers"

# Wait for both processes
wait
