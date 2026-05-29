#!/bin/bash
# Start A股量化分析系统 with public tunnel
# Usage: bash start_public.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Kill old processes
pkill -9 -f "streamlit run app.py" 2>/dev/null || true
pkill -9 -f "localhost.run" 2>/dev/null || true
sleep 2

echo "=== Starting Streamlit on port 8501 ==="
STREAMLIT_SERVER_HEADLESS=true python3 -m streamlit run app.py --server.port 8501 > /tmp/streamlit_public.log 2>&1 &
ST_PID=$!
sleep 4

# Verify Streamlit is up
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q 200; then
    echo "Streamlit is running (PID: $ST_PID)"
else
    echo "ERROR: Streamlit failed to start"
    cat /tmp/streamlit_public.log | tail -10
    exit 1
fi

echo ""
echo "=== Starting public tunnel ==="
echo "Share this URL with others:"
echo ""

# Start tunnel and extract URL
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R 80:localhost:8501 nokey@localhost.run 2>&1 | \
    while IFS= read -r line; do
        echo "$line"
        if echo "$line" | grep -q "tunneled"; then
            URL=$(echo "$line" | grep -o 'https://[^ ]*')
            echo ""
            echo "============================================"
            echo "  PUBLIC URL: $URL"
            echo "  Share this link with others"
            echo "============================================"
        fi
    done
