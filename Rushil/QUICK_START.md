# 🚀 Quick Start Guide

**AI Video Analysis Pipeline** - Powered by Gemma VLM + YOLO X  
*Optimized for RunPod L4 GPU (24GB VRAM, 62GB RAM, 12 vCPUs)*

---

## 📋 Prerequisites

- **Hardware**: NVIDIA L4 GPU with 24GB VRAM (RunPod recommended)
- **Software**: Python 3.8+, CUDA drivers
- **Internet**: Required for initial model downloads

---

## 🛠️ Setup (One-time)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Authenticate with Hugging Face
```bash
python setup_gemma_auth.py
```
*Follow the prompts to authenticate and download the Gemma model*

### 3. Download YOLO Model (if not present)
```bash
# YOLO X model will be downloaded automatically on first run
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
```

---

## 🎬 Video Processing Options

### **Option A: Single Video Analysis**
Perfect for analyzing one video with a specific query.

#### 1. Start VLM Server
```bash
# Keep this terminal open
python scripts/start_server.py
```

#### 2. Run Analysis (New Terminal)
```bash
# Basic usage
python scripts/run_pipeline.py --video your_video.mp4 --query "Find all cats"

# Advanced usage with custom parameters
python scripts/run_pipeline.py \
  --video videos/test_video.mp4 \
  --query "Find all animals and their activities" \
  --fps 2.0 \
  --top-k 10 \
  --batch-size 4
```

#### 3. View Results
```bash
# Check output directory
ls output/
cat output/report.md
```

---

### **Option B: Multi-Video Analysis (Recommended)**
Process multiple videos with custom queries.

#### 1. Upload Videos
```bash
# Interactive upload helper
python scripts/upload_videos.py

# Or manually copy videos to videos/ folder
cp your_videos/*.mp4 videos/
```

#### 2. Configure Queries (Optional)
Edit `video_queries.json`:
```json
{
  "cat_video": "Find all cats and their activities",
  "traffic_video": "Find all vehicles and traffic signs",
  "sports_video": "Identify sports equipment and players"
}
```

#### 3. Run Multi-Video Analysis
```bash
# Interactive selection (recommended)
python scripts/run_multi_video.py

# Or automatic processing of all videos
python scripts/run_concurrent.py
```

#### 4. View Results
```bash
# Each video gets its own output directory
ls output_*/
cat output_cat_video/report.md
cat output_traffic_video/report.md
```

---

### **Option C: Windows Batch Files**
Easy one-click execution for Windows users.

```bash
# Upload videos
upload_videos.bat

# Run multi-video analysis
run_multi_video.bat
```

---

## 🎯 Example Commands

### **Object Detection Queries**
```bash
# Find specific objects
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find all cats"
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find all cars"
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find all people"
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find all animals"
```

### **Activity & Scene Analysis**
```bash
# Detect activities
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find people doing activities"
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Detect motion and movement"
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Identify outdoor activities"

# Scene understanding
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Describe the indoor scene"
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find natural elements"
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Identify sports or games"
```

### **Advanced Analysis**
```bash
# High-frequency analysis (more frames)
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find all objects" --fps 3.0

# More detailed results
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find all cats" --top-k 15

# Optimized for GPU memory
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find all cats" --batch-size 2
```

---

## 📊 Understanding Output

### **Report Structure**
```
output_test_video/
├── report.json          # Machine-readable results
├── report.md            # Human-readable report
├── frame_1.50s.jpg      # Annotated frames with detections
├── frame_3.20s.jpg
├── test_video1.50s_cat_0.85_0.jpg  # Cropped objects
└── test_video3.20s_cat_0.92_1.jpg
```

### **Report Contents**
- **Final Answer**: Summary of findings
- **Top Frames**: Most relevant timestamps with reasoning
- **Evidence Table**: Timestamps, rationales, detections, and crops
- **Visual Evidence**: Annotated frames and object crops

---

## ⚡ Performance Tips

### **GPU Optimization**
```bash
# For L4 GPU (24GB VRAM) - Default settings work well
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find all cats"

# If memory issues occur, reduce batch size
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find all cats" --batch-size 4

# For faster processing, reduce FPS
python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find all cats" --fps 0.5
```

### **Multi-Video Optimization**
- Process 2-4 videos concurrently on L4 GPU
- Use SSD storage for better I/O performance
- Monitor GPU memory usage during processing

---

## 🔧 Troubleshooting

### **Common Issues**

1. **"No video files found"**
   ```bash
   # Ensure videos are in the right location
   ls videos/
   # Or use upload helper
   python scripts/upload_videos.py
   ```

2. **"Model not available"**
   ```bash
   # Re-authenticate with Hugging Face
   python setup_gemma_auth.py
   ```

3. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python scripts/run_pipeline.py --video videos/test_video.mp4 --query "Find cats" --batch-size 2
   ```

4. **Server Connection Failed**
   ```bash
   # Make sure VLM server is running
   python scripts/start_server.py
   ```

### **Performance Issues**
- **Slow Processing**: Reduce `--fps` parameter
- **Memory Issues**: Reduce `--batch-size` parameter  
- **Low Accuracy**: Increase `--top-k` parameter

---

## 🐳 Docker Alternative

### **Build and Run**
```bash
# Build container
docker-compose up --build

# Run analysis in container
docker-compose exec gemma-vlm-server python3 scripts/run_pipeline.py \
  --video /app/videos/test_video.mp4 \
  --query "Find all cats"
```

---

## 📁 File Organization

### **Input Structure**
```
videos/                    # 📹 Put your videos here
├── test_video.mp4         # Sample video
├── cat_video.mp4          # Your videos
├── traffic_video.mp4
└── sports_video.mp4

video_queries.json         # Custom queries for each video
```

### **Output Structure**
```
output_test_video/         # Results for test_video.mp4
output_cat_video/          # Results for cat_video.mp4
output_traffic_video/      # Results for traffic_video.mp4
```

---

## 🎉 Next Steps

1. **Upload Your Videos**: Use `python scripts/upload_videos.py`
2. **Customize Queries**: Edit `video_queries.json`
3. **Run Analysis**: Use `python scripts/run_multi_video.py`
4. **View Results**: Check individual output directories

---
