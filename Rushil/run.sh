

# Video Analysis Pipeline - Main execution script

set -e

echo "=== Video Analysis Pipeline ==="
echo "Starting services..."

if command -v docker-compose &> /dev/null; then
    echo "Using Docker Compose..."
    
 
    docker-compose up --build -d
    

    echo "Waiting for VLM server to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "VLM server is ready!"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        echo "Error: VLM server failed to start"
        docker-compose logs gemma-vlm-server
        exit 1
    fi
    
  
    echo "Running video analysis pipeline..."
    docker-compose exec gemma-vlm-server python3 scripts/run_pipeline.py "$@"
    
else
    echo "Docker Compose not available, running locally..."
    

    if ! curl -s http://localhost:8000/health > /dev/null; then
        echo "Starting VLM server in background..."
        python3 scripts/start_server.py &
        SERVER_PID=$!
        
      
        echo "Waiting for VLM server to be ready..."
        timeout=60
        while [ $timeout -gt 0 ]; do
            if curl -s http://localhost:8000/health > /dev/null; then
                echo "VLM server is ready!"
                break
            fi
            sleep 2
            timeout=$((timeout - 2))
        done
        
        if [ $timeout -le 0 ]; then
            echo "Error: VLM server failed to start"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
    fi
    

echo "Running video analysis pipeline..."
if [[ "$1" == "--concurrent" ]]; then
  shift
  python3 scripts/run_concurrent.py "$@"
else
  python3 scripts/run_pipeline.py "$@"
fi
    
    # Cleanup
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping VLM server..."
        kill $SERVER_PID 2>/dev/null || true
    fi
fi

echo "Pipeline execution completed!"
