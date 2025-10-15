# Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VIDEO ANALYSIS PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────┐
│   Video Input   │───▶│  Frame Extract   │───▶│  LLM Analysis   │───▶│ Frame Select │
│   (MP4/MKV)     │    │     (1 fps)      │    │  (Gemma VLM)    │    │   (Top-K)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────┘
                                │                        │                    │
                                ▼                        ▼                    ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────┐
│  Report Gen     │◀───│  Object Crops    │◀───│  YOLO Detection │◀───│ Frame Cache │
│ (JSON + MD)     │    │   (BBox Crop)    │    │   + BBoxes      │    │   (Dedupe)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────┘
```

## Component Details

### 1. Gemma VLM Server
```
┌─────────────────────────────────────────────────────────────────┐
│                    Gemma VLM Server (FastAPI)                   │
├─────────────────────────────────────────────────────────────────┤
│  • Model: google/paligemma-3b-pt-224                           │
│  • Device: CUDA/CPU                                             │
│  • Batch Processing: 8 images/batch                             │
│  • Endpoints:                                                    │
│    - POST /analyze (images + query → analysis)                  │
│    - GET /health (latency + GPU metrics)                        │
│    - GET /metrics (system performance)                          │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Video Processing Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│                    Video Processor                              │
├─────────────────────────────────────────────────────────────────┤
│  • Frame Extraction: ffmpeg (1 fps)                            │
│  • Batch Analysis: Async HTTP calls                            │
│  • Frame Selection: Top-K by relevance score                   │
│  • Object Detection: YOLO v8                                   │
│  • Report Generation: JSON + Markdown                          │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Data Flow
```
Video File (MP4/MKV)
    │
    ▼
┌─────────────────┐
│  Frame Extract  │ ── ffmpeg ──► [Frame1, Frame2, ..., FrameN]
└─────────────────┘     (1 fps)     (timestamps + RGB arrays)
    │
    ▼
┌─────────────────┐
│ Batch Process   │ ── base64 encode ──► HTTP POST ──► VLM Server
└─────────────────┘     (batch_size=8)     /analyze
    │
    ▼
┌─────────────────┐
│ Relevance Score │ ── sort by score ──► [Top-K Frames]
└─────────────────┘     (relevance)        (most relevant)
    │
    ▼
┌─────────────────┐
│ YOLO Detection  │ ── object detection ──► [Bounding Boxes]
└─────────────────┘     (yolov8x.pt)        (label + confidence)
    │
    ▼
┌─────────────────┐
│ Object Cropping │ ── crop bbox ──► [Object Images]
└─────────────────┘     (save files)        (timestamped crops)
    │
    ▼
┌─────────────────┐
│ Report Generate │ ── JSON + MD ──► [Final Reports]
└─────────────────┘     (structured)        (machine + human readable)
```

## Performance Characteristics

### Processing Stages
```
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ Extraction  │ LLM Analysis│ Frame Select│ YOLO Detect │ Report Gen  │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ 10-30 fps   │ 0.5-2 fps   │ <1ms        │ 20-50 fps   │ <100ms      │
│ (I/O bound) │ (GPU bound) │ (CPU)       │ (GPU opt)   │ (I/O)       │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

### Memory Usage
```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Gemma Model     │ Frame Buffers   │ YOLO Model      │ Total System    │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ ~3-8 GB VRAM    │ ~100 MB RAM     │ ~50 MB RAM      │ ~4-10 GB        │
│ (3B parameters) │ (per 100 frames)│ (yolov8x.pt)    │ (depends on GPU)│
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

## Deployment Architecture

### Docker Setup
```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                   │
│  │ gemma-vlm-server│    │   Host Volumes  │                   │
│  │                 │    │                 │                   │
│  │ • Port: 8000    │    │ • ./output/     │                   │
│  │ • GPU Support   │    │ • ./models/     │                   │
│  │ • Health Check  │    │ • ./tests/      │                   │
│  │ • Auto Restart  │    │                 │                   │
│  └─────────────────┘    └─────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### API Interaction
```
┌─────────────────┐    HTTP/JSON    ┌─────────────────┐
│  Client Script  │ ───────────────►│  VLM Server     │
│                 │                 │                 │
│ • Video Input   │                 │ • Model Serving │
│ • Query Text    │                 │ • Batch Process │
│ • Output Config │                 │ • GPU Monitor   │
└─────────────────┘                 └─────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────┐                 ┌─────────────────┐
│ Local Pipeline  │                 │ Remote Analysis │
│                 │                 │                 │
│ • Frame Extract │                 │ • Scene Under-  │
│ • Result Process│                 │   standing      │
│ • Report Gen    │                 │ • Relevance     │
└─────────────────┘                 │   Scoring       │
                                    └─────────────────┘
```

## Error Handling Flow

```
┌─────────────────┐
│   Input Video   │
└─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐
│ Model Load OK?  │───▶│   Mock Mode     │
└─────────────────┘    │ (Fallback)      │
         │ Yes          └─────────────────┘
         ▼
┌─────────────────┐    ┌─────────────────┐
│ GPU Available?  │───▶│   CPU Mode      │
└─────────────────┘    │ (Slower)        │
         │ Yes          └─────────────────┘
         ▼
┌─────────────────┐    ┌─────────────────┐
│ ffmpeg OK?      │───▶│ OpenCV Fallback │
└─────────────────┘    │ (Compatible)    │
         │ Yes          └─────────────────┘
         ▼
┌─────────────────┐
│ Normal Pipeline │
└─────────────────┘
```

## Monitoring & Metrics

```
┌─────────────────────────────────────────────────────────────────┐
│                    System Metrics                              │
├─────────────────────────────────────────────────────────────────┤
│  • GPU Utilization: torch.cuda.utilization()                  │
│  • Memory Usage: psutil.virtual_memory()                      │
│  • Processing Times: Per-stage timing                          │
│  • Throughput: Frames per second                               │
│  • Error Rates: Failed requests / total                       │
│  • Health Status: /health endpoint                             │
└─────────────────────────────────────────────────────────────────┘
```
