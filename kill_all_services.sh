#!/bin/bash

echo "🛑 Stopping all ThriftAssist services..."

# Kill all uvicorn processes
echo "💀 Killing uvicorn processes..."
pkill -f "uvicorn.*main:app" 2>/dev/null || echo "No uvicorn processes found"

# Kill processes on common ports
for port in 8080 8081 8000 5000; do
    pids=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        echo "💀 Killing processes on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null
    fi
done

echo "✅ All services stopped"
