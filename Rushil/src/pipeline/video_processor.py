#!/usr/bin/env python3
"""
Video Processing Pipeline
Main pipeline that processes videos through frame extraction, LLM analysis, and YOLO detection
"""

import os
import json
import time
import logging
import asyncio
import aiohttp
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import hashlib

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io

# Try to import ffmpeg, fall back to OpenCV if not available
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    print("Warning: ffmpeg-python not available, using OpenCV fallback")

logger = logging.getLogger(__name__)

@dataclass
class FrameAnalysis:
    """Analysis result for a single frame"""
    timestamp: float
    caption: str
    entities: List[str]
    reasoning: str
    relevance_score: float
    frame_path: Optional[str] = None
    detections: Optional[List[Dict]] = None
    crop_paths: Optional[List[str]] = None

class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self, 
                 vlm_server_url: str = "http://localhost:8000",
                 fps_rate: float = 1.0, 
                 top_k: int = 5, 
                 batch_size: int = 8, 
                 max_workers: int = 4,
                 yolo_model: str = "yolov8x.pt"):
        """
        Initialize the video processor
        
        Args:
            vlm_server_url: URL of the Gemma VLM server
            fps_rate: Frames per second to extract
            top_k: Number of top frames to select
            batch_size: Batch size for LLM analysis
            max_workers: Maximum number of worker threads
            yolo_model: YOLO model to use for detection
        """
        self.vlm_server_url = vlm_server_url
        self.fps_rate = fps_rate
        self.top_k = top_k
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timings = {
            'extraction': [],
            'llm_analysis': [],
            'detection': [],
            'total': []
        }
        
        # Initialize YOLO
        try:
            logger.info(f"Loading YOLO model: {yolo_model}")
            self.yolo = YOLO(yolo_model)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.warning(f"YOLO not available - detection disabled: {e}")
            self.yolo = None
        
        # Frame cache to avoid reprocessing identical frames
        self.frame_cache = {}
    
    def extract_frames(self, video_path: str) -> List[Tuple[float, np.ndarray]]:
        """Extract frames at specified FPS using ffmpeg or OpenCV"""
        start_time = time.time()
        frames = []
        
        # Try ffmpeg first if available
        if FFMPEG_AVAILABLE:
            try:
                logger.info(f"Extracting frames from {video_path} at {self.fps_rate} fps using ffmpeg")
                
                # Probe video to get FPS
                probe = ffmpeg.probe(video_path)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                video_fps = eval(video_info['r_frame_rate'])
                
                logger.info(f"Video FPS: {video_fps}")
                
                # Use ffmpeg to extract frames at specified rate
                out, _ = (
                    ffmpeg
                    .input(video_path)
                    .filter('fps', fps=self.fps_rate)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True, quiet=True)
                )
                
                # Get video dimensions
                width = video_info['width']
                height = video_info['height']
                
                # Calculate number of frames
                frame_size = width * height * 3
                num_frames = len(out) // frame_size
                
                logger.info(f"Extracted {num_frames} frames")
                
                # Convert bytes to numpy arrays
                for i in range(num_frames):
                    timestamp = i / self.fps_rate
                    frame_start = i * frame_size
                    frame_end = frame_start + frame_size
                    frame_bytes = out[frame_start:frame_end]
                    
                    # Reshape to image format
                    frame = np.frombuffer(frame_bytes, np.uint8).reshape((height, width, 3))
                    frames.append((timestamp, frame))
                    
                self.timings['extraction'].append(time.time() - start_time)
                logger.info(f"Extracted {len(frames)} frames in {self.timings['extraction'][-1]:.2f}s")
                return frames
                
            except Exception as e:
                logger.warning(f"FFmpeg extraction failed, falling back to OpenCV: {e}")
        
        # Fallback to OpenCV
        frames = self._extract_frames_opencv(video_path)
        self.timings['extraction'].append(time.time() - start_time)
        logger.info(f"Extracted {len(frames)} frames in {self.timings['extraction'][-1]:.2f}s")
        return frames
    
    def _extract_frames_opencv(self, video_path: str) -> List[Tuple[float, np.ndarray]]:
        """Fallback frame extraction using OpenCV"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return frames
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / self.fps_rate) if self.fps_rate > 0 else 1
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((timestamp, frame_rgb))
            
            frame_count += 1
        
        cap.release()
        logger.info(f"OpenCV extracted {len(frames)} frames")
        return frames
    
    async def analyze_frames_batch(self, frames: List[Tuple[float, np.ndarray]], query: str) -> List[FrameAnalysis]:
        """Analyze frames in batches using the VLM server"""
        start_time = time.time()
        results = []
        
        logger.info(f"Analyzing {len(frames)} frames in batches of {self.batch_size}")
        
        # Process frames in batches
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            batch_results = await self._analyze_single_batch(batch_frames, query)
            results.extend(batch_results)
        
        self.timings['llm_analysis'].append(time.time() - start_time)
        logger.info(f"Analysis completed in {self.timings['llm_analysis'][-1]:.2f}s")
        return results
    
    async def _analyze_single_batch(self, frames: List[Tuple[float, np.ndarray]], query: str) -> List[FrameAnalysis]:
        """Analyze a single batch of frames"""
        timestamps = [ts for ts, _ in frames]
        images = [Image.fromarray(frame) for _, frame in frames]
        
        # Convert images to base64
        image_b64s = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            image_b64s.append(img_b64)
        
        # Call VLM server
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "images": image_b64s,
                    "query": query
                }
                
                async with session.post(
                    f"{self.vlm_server_url}/analyze",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        batch_results = []
                        
                        for ts, analysis_result in zip(timestamps, result['results']):
                            frame_analysis = FrameAnalysis(
                                timestamp=ts,
                                caption=analysis_result['caption'],
                                entities=analysis_result['entities'],
                                reasoning=analysis_result['reasoning'],
                                relevance_score=analysis_result['relevance_score']
                            )
                            batch_results.append(frame_analysis)
                        
                        return batch_results
                    else:
                        logger.error(f"VLM server error: {response.status}")
                        return [FrameAnalysis(
                            timestamp=ts,
                            caption="Analysis failed",
                            entities=[],
                            reasoning="Server error",
                            relevance_score=0.0
                        ) for ts in timestamps]
                        
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return [FrameAnalysis(
                timestamp=ts,
                caption=f"Analysis error: {str(e)}",
                entities=[],
                reasoning="Network or processing error",
                relevance_score=0.0
            ) for ts in timestamps]
    
    def select_top_frames(self, analyses: List[FrameAnalysis]) -> List[FrameAnalysis]:
        """Select top-K most relevant frames"""
        sorted_analyses = sorted(analyses, key=lambda x: x.relevance_score, reverse=True)
        selected = sorted_analyses[:self.top_k]
        
        logger.info(f"Selected top {len(selected)} frames with relevance scores: "
                   f"{[f'{a.relevance_score:.3f}' for a in selected]}")
        
        return selected
    
    def run_detection(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLO detection on a single frame"""
        if self.yolo is None:
            return []
        
        start_time = time.time()
        detections = []
        
        try:
            results = self.yolo(frame, verbose=False)
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        label = self.yolo.names[cls]
                        
                        detections.append({
                            'label': label,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
        
        self.timings['detection'].append(time.time() - start_time)
        return detections
    
    def crop_detections(self, frame: np.ndarray, detections: List[Dict], 
                       output_dir: Path, timestamp: float) -> List[str]:
        """Crop detected objects and save them"""
        crop_paths = []
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            confidence = det['confidence']
            
            # Crop the detection
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                # Create filename
                crop_filename = f"{timestamp:.2f}s_{label}_{confidence:.2f}_{i}.jpg"
                crop_path = output_dir / crop_filename
                
                # Save crop
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_path), crop_rgb)
                crop_paths.append(str(crop_path))
        
        return crop_paths
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        img = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['label']} {det['confidence']:.2f}"
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return img
    
    async def process_video(self, video_path: str, query: str, output_dir: str = "output") -> Dict[str, Any]:
        """Main pipeline execution"""
        pipeline_start = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Query: '{query}'")
        logger.info(f"Output directory: {output_dir}")
        
        # Extract frames
        frames = self.extract_frames(video_path)
        if not frames:
            logger.error("No frames extracted, stopping pipeline")
            return {}
        
        # Analyze frames
        analyses = await self.analyze_frames_batch(frames, query)
        
        # Select top frames
        selected_analyses = self.select_top_frames(analyses)
        
        # Create frame lookup
        frame_dict = dict(frames)
        
        # Process selected frames
        logger.info(f"Processing {len(selected_analyses)} selected frames")
        
        for analysis in selected_analyses:
            timestamp = analysis.timestamp
            if timestamp in frame_dict:
                frame_data = frame_dict[timestamp]
                
                # Run detection
                detections = self.run_detection(frame_data)
                analysis.detections = detections
                
                # Crop detections
                if detections:
                    crop_paths = self.crop_detections(frame_data, detections, output_path, timestamp)
                    analysis.crop_paths = crop_paths
                
                # Save frame with detections
                frame_filename = f"frame_{timestamp:.2f}s.jpg"
                frame_save_path = output_path / frame_filename
                
                if detections:
                    image_to_save = self.draw_detections(frame_data, detections)
                else:
                    image_to_save = frame_data
                
                # Convert RGB to BGR for OpenCV
                image_bgr = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_save_path), image_bgr)
                analysis.frame_path = str(frame_save_path)
        
        # Generate reports
        report = self.generate_report(query, selected_analyses, video_path, len(frames))
        
        # Save JSON report
        with open(output_path / "report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate Markdown report
        self.generate_markdown_report(report, output_path / "report.md")
        
        self.timings['total'].append(time.time() - pipeline_start)
        logger.info(f"Pipeline completed in {self.timings['total'][-1]:.2f}s")
        
        return report
    
    def generate_report(self, query: str, analyses: List[FrameAnalysis], 
                       video_path: str, total_frames: int) -> Dict[str, Any]:
        """Generate structured JSON report"""
        # Create final answer
        if analyses and analyses[0].relevance_score > 0.1:
            top_frame = analyses[0]
            answer = f"Based on the analysis, the most relevant content appears at {top_frame.timestamp:.2f}s. " \
                    f"The scene shows: {top_frame.caption}"
        else:
            answer = "No relevant content found for the given query."
        
        # Calculate performance metrics
        performance = {
            'total_frames_extracted': total_frames,
            'frames_analyzed': len(analyses),
            'frames_selected': len(analyses),
            'avg_extraction_time_s': np.mean(self.timings['extraction']) if self.timings['extraction'] else 0,
            'total_llm_time_s': np.sum(self.timings['llm_analysis']) if self.timings['llm_analysis'] else 0,
            'avg_detection_time_s': np.mean(self.timings['detection']) if self.timings['detection'] else 0,
            'total_pipeline_time_s': np.sum(self.timings['total']) if self.timings['total'] else 0,
            'processing_fps': total_frames / max(np.sum(self.timings['total']), 0.001)
        }
        
        return {
            'query': query,
            'video_path': video_path,
            'timestamp': time.time(),
            'final_answer': answer,
            'top_frames': [asdict(analysis) for analysis in analyses],
            'performance': performance,
            'limits': {
                'max_frames_analyzed': len(analyses),
                'processing_time_limit': 'No limit set',
                'memory_usage': 'Not tracked'
            }
        }
    
    def generate_markdown_report(self, report: Dict[str, Any], output_path: Path):
        """Generate human-readable Markdown report"""
        with open(output_path, 'w') as f:
            f.write(f"# Video Analysis Report\n\n")
            f.write(f"**Query:** {report['query']}\n\n")
            f.write(f"**Video:** {report['video_path']}\n\n")
            f.write(f"**Analysis Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['timestamp']))}\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"{report['final_answer']}\n\n")
            
            f.write(f"## Detailed Analysis\n\n")
            f.write(f"### Top Frames\n\n")
            
            for i, frame in enumerate(report['top_frames'], 1):
                f.write(f"#### Frame {i}: {frame['timestamp']:.2f}s (Relevance: {frame['relevance_score']:.3f})\n\n")
                f.write(f"**Caption:** {frame['caption']}\n\n")
                f.write(f"**Entities:** {', '.join(frame['entities']) if frame['entities'] else 'None'}\n\n")
                f.write(f"**Reasoning:** {frame['reasoning']}\n\n")
                
                if frame['detections']:
                    f.write(f"**Detected Objects:**\n")
                    for det in frame['detections']:
                        f.write(f"- {det['label']} (confidence: {det['confidence']:.2f})\n")
                    f.write(f"\n")
                
                if frame['frame_path']:
                    f.write(f"**Frame Image:** `{frame['frame_path']}`\n\n")
                
                if frame['crop_paths']:
                    f.write(f"**Object Crops:**\n")
                    for crop_path in frame['crop_paths']:
                        f.write(f"- `{crop_path}`\n")
                    f.write(f"\n")
            
            f.write(f"## Performance Metrics\n\n")
            perf = report['performance']
            f.write(f"- **Total Frames Extracted:** {perf['total_frames_extracted']}\n")
            f.write(f"- **Frames Analyzed:** {perf['frames_analyzed']}\n")
            f.write(f"- **Processing FPS:** {perf['processing_fps']:.2f}\n")
            f.write(f"- **Total Pipeline Time:** {perf['total_pipeline_time_s']:.2f}s\n")
            f.write(f"- **LLM Analysis Time:** {perf['total_llm_time_s']:.2f}s\n")
            f.write(f"- **Detection Time:** {perf['avg_detection_time_s']:.2f}s per frame\n\n")
