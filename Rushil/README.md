# 🎬 AI Video Analysis Pipeline

A powerful concurrent video analysis system that combines **Gemma Vision-Language Model (VLM)** with **YOLO object detection** to analyze video content and answer natural language queries about what's happening in the footage.

## 🚀 Features

- **Concurrent Processing**: Analyze multiple videos simultaneously using AsyncIO and ThreadPoolExecutor
- **Vision-Language Understanding**: Powered by Google's Gemma-3B model for intelligent video analysis
- **Object Detection**: YOLOv8 integration for precise object identification and bounding box detection
- **Natural Language Queries**: Ask questions about video content in plain English
- **Comprehensive Reports**: Generate both JSON and Markdown reports with visual evidence
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **Docker Support**: Containerized deployment for easy setup

## 💻 System Requirements

### Recommended Hardware (RunPod Configuration)
- **GPU**: NVIDIA L4 GPU with 24GB VRAM
- **RAM**: 62GB System Memory
- **CPU**: 12 vCPUs
- **Storage**: SSD recommended for video processing

### Software Requirements
- Python 3.8+
- CUDA-compatible GPU drivers
- Docker (optional)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd video-analysis-pipeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Authenticate with Hugging Face (One-time setup)
```bash
python setup_gemma_auth.py
```

## 🎯 Quick Start

### Single Video Analysis
```bash
# Start the VLM server (keep terminal open)
python scripts/start_server.py

# In a new terminal, run analysis
python scripts/run_pipeline.py --video test_video.mp4 --query "Find all cats"
```

### Concurrent Multi-Video Analysis
```bash
# Process all videos in videos folder automatically
python scripts/run_concurrent.py

# Interactive multi-video selection with custom queries
python scripts/run_multi_video.py
```

### Video Management
```bash
# Upload and organize videos
python scripts/upload_videos.py

# Using the Helper Script
chmod +x run.sh
./run.sh --concurrent
```

## 📊 Example Queries

### Object Detection
```bash
# Find specific objects
python scripts/run_pipeline.py --video test_video.mp4 --query "Find all cats"
python scripts/run_pipeline.py --video test_video.mp4 --query "Find all cars"
python scripts/run_pipeline.py --video test_video.mp4 --query "Find all people"
python scripts/run_pipeline.py --video test_video.mp4 --query "Find all animals"
```

### Motion & Activity Analysis
```bash
# Detect movement and activities
python scripts/run_pipeline.py --video test_video.mp4 --query "Find moving objects"
python scripts/run_pipeline.py --video test_video.mp4 --query "Detect motion and movement"
python scripts/run_pipeline.py --video test_video.mp4 --query "Identify suspicious activity"
```

### Scene Understanding
```bash
# Analyze scenes and contexts
python scripts/run_pipeline.py --video test_video.mp4 --query "Describe what's happening"
python scripts/run_pipeline.py --video test_video.mp4 --query "Find outdoor activities"
python scripts/run_pipeline.py --video test_video.mp4 --query "Identify indoor scenes"
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the container
docker-compose up --build

# Run analysis in container
docker-compose exec gemma-vlm-server python3 scripts/run_pipeline.py \
  --video /app/test_video.mp4 \
  --query "Find all cats"
```

### Docker Configuration
The included `docker-compose.yml` and `Dockerfile` provide:
- Pre-configured environment with all dependencies
- GPU access for CUDA acceleration
- Volume mounting for input/output files
- Optimized for RunPod deployment

## 📁 Project Structure

```
├── src/
│   ├── models/
│   │   └── gemma_server.py          # Gemma VLM model wrapper
│   └── pipeline/
│       ├── concurrent_pipeline.py   # Main concurrent processing engine
│       └── video_processor.py       # Video processing utilities
├── scripts/
│   ├── run_concurrent.py           # Automatic multi-video analysis
│   ├── run_multi_video.py          # Interactive video selection
│   ├── run_pipeline.py             # Single video analysis
│   ├── start_server.py             # VLM server startup
│   └── upload_videos.py            # Video upload helper
├── videos/                         # 📹 Upload your videos here
│   ├── test_video.mp4              # Sample video
│   ├── your_video1.mp4             # Your uploaded videos
│   └── your_video2.mp4
├── tests/
│   ├── run_tests.py                # Test suite
│   └── test_queries.json           # Sample queries
├── output/                         # Analysis results
├── video_queries.json              # Custom video queries config
├── test_video.mp4                  # Fallback sample video
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Docker configuration
├── Dockerfile                      # Container definition
└── README.md                       # This file
```

## 📹 Video Management

### Upload Videos
1. **Using Upload Helper**:
   ```bash
   python scripts/upload_videos.py
   ```

2. **Manual Upload**:
   - Place video files (.mp4, .avi, .mov, .mkv, .webm, .flv) in the `videos/` folder
   - Supported formats: MP4, AVI, MOV, MKV, WebM, FLV, WMV, M4V

### Custom Queries
Edit `video_queries.json` to customize analysis queries:
```json
{
  "your_video": "Find all cars and traffic signs",
  "sports_clip": "Identify sports equipment and players",
  "nature_video": "Find animals and natural elements"
}
```

## 📈 Performance Optimization

### GPU Memory Management
- The system automatically detects GPU availability and adjusts batch sizes
- Uses mixed precision (bfloat16) for optimal memory usage on L4 GPUs
- Concurrent processing maximizes GPU utilization

### Processing Configuration
```python
# Adjustable parameters in concurrent_pipeline.py
fps_rate = 1.0        # Frames per second to analyze
batch_size = 8        # Batch size for VLM processing
top_k = 5            # Number of top frames to return
max_workers = 4      # Thread pool size
```

## 📊 Output Reports

### JSON Report (`report.json`)
```json
{
  "query": "Find all cats",
  "video": "test_video.mp4",
  "final_answer": "Most relevant content at 2.50s: 'A cat sitting on a chair'",
  "top_frames": [
    {
      "timestamp": 2.5,
      "caption": "A cat sitting on a chair",
      "entities": ["cat", "chair"],
      "reasoning": "Found 1 matching keywords.",
      "relevance_score": 0.8,
      "detections": [{"label": "cat", "confidence": 0.85}],
      "frame_path": "output_test_video/frame_at_2.50s.jpg"
    }
  ]
}
```

### Markdown Report (`report.md`)
- Human-readable summary with visual evidence
- Timestamped findings with confidence scores
- Embedded images showing detected objects
- Evidence table with rationales

## 🔧 Configuration

### Model Settings
- **Default VLM**: `google/paligemma-3b-pt-224`
- **Default YOLO**: `yolov8x.pt` (high accuracy model - optimized for L4 GPU)
- **Device**: Auto-detects CUDA availability

### Environment Variables
```bash
# Optional: Override default settings
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/path/to/huggingface/cache
export TRANSFORMERS_CACHE=/path/to/transformers/cache
```

## 🧪 Testing

### Run Test Suite
```bash
python tests/run_tests.py
```

### Test with Sample Queries
```bash
# Test various query types
python scripts/run_pipeline.py --video test_video.mp4 --query "Find all cats"
python scripts/run_pipeline.py --video test_video.mp4 --query "Describe the scene"
python scripts/run_pipeline.py --video test_video.mp4 --query "Find moving objects"
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in pipeline configuration
   - Reduce batch size or processing parameters

2. **Model Loading Failed**
   - Ensure Hugging Face authentication is complete
   - Check internet connection for model downloads
   - Verify sufficient disk space for model cache

3. **Video Processing Errors**
   - Ensure video file is in supported format (MP4, AVI, MOV)
   - Check file permissions and path accessibility
   - Verify OpenCV installation with video codec support

### Performance Tips
- Use SSD storage for faster video I/O
- Ensure adequate GPU cooling for sustained processing
- Monitor GPU memory usage during concurrent operations
- Consider processing shorter video segments for very large files

## 📝 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Analysis Endpoint
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_path": "test_video.mp4", "query": "Find all cats"}'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemma Team** for the Vision-Language Model
- **Ultralytics** for YOLO object detection
- **Hugging Face** for model hosting and transformers library
- **RunPod** for GPU cloud infrastructure support

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section above
- Review the test suite for usage examples

---

**Optimized for RunPod L4 GPU with 24GB VRAM, 62GB RAM, and 12 vCPUs**
