#!/usr/bin/env python3
"""
Concurrent Video Analysis Pipeline (AsyncIO + Threads)
Adapted from provided Colab code to project structure.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from ultralytics import YOLO

# Optional ffmpeg-python; fallback to OpenCV timings if unavailable
try:
    import ffmpeg  # type: ignore
    FFMPEG_AVAILABLE = True
except Exception:
    FFMPEG_AVAILABLE = False

logger = logging.getLogger(__name__)


# ===================== Data Classes and Model Server =====================
@dataclass
class FrameAnalysis:
    timestamp: float
    caption: str
    entities: List[str]
    reasoning: str
    relevance_score: float
    frame_path: Optional[str] = None
    detections: Optional[List[Dict]] = None
    crop_paths: Optional[List[str]] = None


class GemmaVLMServer:
    def __init__(self, model_name: str = "google/paligemma-3b-pt-224", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Gemma VLM on {self.device}")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
                device_map="auto"
            )
            if self.device != 'cuda':
                self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model, self.processor = None, None
            logger.warning("Model not loaded - analysis will return errors")

    def analyze_image_batch_sync(self, images: List[Image.Image], query: str) -> List[Dict[str, Any]]:
        if self.model is None:
            return [{
                'caption': 'Model not available - please authenticate and download Gemma',
                'entities': [],
                'reasoning': 'Gemma model not loaded',
                'relevance_score': 0.0
            } for _ in images]

        try:
            model_prompt_template = f"Based on the query '{query}', describe the contents of this image."
            prompts = [f"<image>\n{model_prompt_template}"] * len(images)
            inputs = self.processor(
                images=images, text=prompts, return_tensors="pt", padding=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100)
            raw_responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
            cleaned_responses = [resp.split(model_prompt_template)[-1].strip() for resp in raw_responses]
            return [self._parse_response(resp, query) for resp in cleaned_responses]
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return [{'caption': 'Analysis failed', 'entities': [], 'reasoning': str(e), 'relevance_score': 0.0} for _ in images]

    def _parse_response(self, response: str, query: str) -> Dict[str, Any]:
        words = set(response.lower().split())
        query_words = set(query.lower().split())
        common_words = words.intersection(query_words)
        relevance = len(common_words) / (len(query_words) + 1e-6)
        entities = [w for w in response.split() if len(w) > 4 and w.lower() not in query_words][:5]
        return {'caption': response[:250], 'entities': entities, 'reasoning': f"Found {len(common_words)} matching keywords.", 'relevance_score': relevance}


# ===================== Markdown Report Generator =====================
def generate_markdown_report(report: Dict[str, Any], output_dir: str):
    Path(output_dir).mkdir(exist_ok=True)
    md_lines = []
    md_lines.append(f"# Video Analysis Report\n")
    md_lines.append(f"## Query\n*{report.get('query', 'N/A')}*\n")
    md_lines.append(f"## Final Answer\n{report.get('final_answer', 'N/A')}\n")

    summary = report.get('final_answer', 'N/A')
    top_frames = report.get('top_frames', [])
    if top_frames:
        timestamps = ', '.join([f"{frame['timestamp']:.2f}s" for frame in top_frames])
        summary += f" Evidence found at {timestamps}."
    md_lines.append(f"### Summary\n{summary}\n")

    md_lines.append("## Evidence Table")
    md_lines.append("| Timestamp | Rationale | Detections | Crops |")
    md_lines.append("|---|---|---|---|")

    for frame in top_frames:
        timestamp_str = f"{frame.get('timestamp', 0.0):.2f}s"
        rationale = frame.get('reasoning', '')
        detections = frame.get('detections', [])
        det_labels = ', '.join([d['label'] for d in detections]) if detections else ""
        crop_paths = frame.get('crop_paths', [])
        crops = ' '.join([f"![crop]({Path(crop).name})" for crop in crop_paths])
        md_lines.append(f"| {timestamp_str} | {rationale} | {det_labels} | {crops} |")

    md_path = Path(output_dir) / "report.md"
    with open(md_path, "w") as f:
        f.write('\n'.join(md_lines))
    logger.info(f"Markdown report saved to: {md_path}")


# ===================== Asynchronous Video Processing Pipeline =====================
class VideoPipeline:
    def __init__(self, vlm_server: GemmaVLMServer, executor: ThreadPoolExecutor, fps_rate: float = 1.0, top_k: int = 5, batch_size: int = 8, yolo_version: str = 'yolov8x.pt'):
        self.vlm_server = vlm_server
        self.executor = executor
        self.fps_rate, self.top_k, self.batch_size = fps_rate, top_k, batch_size
        self.timings = {'extraction': [], 'llm_analysis': [], 'detection': [], 'total': []}
        try:
            self.yolo = YOLO(yolo_version)
        except Exception as e:
            logger.warning(f"YOLO not available: {e}")
            self.yolo = None

    async def _run_in_executor(self, func, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, func, *args)

    async def extract_frames(self, video_path: str) -> List[Tuple[float, np.ndarray]]:
        def sync_extract():
            start_time = time.time()
            frames = []
            try:
                if FFMPEG_AVAILABLE:
                    probe = ffmpeg.probe(video_path)
                    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                    fps = eval(video_info['r_frame_rate'])
                else:
                    cap_probe = cv2.VideoCapture(video_path)
                    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
                    cap_probe.release()

                cap = cv2.VideoCapture(video_path)
                frame_interval = max(1, int(fps / self.fps_rate))
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_count % frame_interval == 0:
                        frames.append((frame_count / fps, frame))
                    frame_count += 1
                cap.release()
            except Exception as e:
                logger.error(f"Frame extraction failed for {video_path}: {e}")
            self.timings['extraction'].append(time.time() - start_time)
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames
        return await self._run_in_executor(sync_extract)

    async def analyze_frames(self, frames: List[Tuple[float, np.ndarray]], query: str) -> List[FrameAnalysis]:
        start_time = time.time()
        results = []
        for i in range(0, len(frames), self.batch_size):
            batch_tuples = frames[i:i+self.batch_size]
            timestamps = [ts for ts, _ in batch_tuples]
            images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for _, f in batch_tuples]
            batch_analysis_results = await self._run_in_executor(
                self.vlm_server.analyze_image_batch_sync, images, query
            )
            for ts, analysis_result in zip(timestamps, batch_analysis_results):
                results.append(FrameAnalysis(timestamp=ts, **analysis_result))
        self.timings['llm_analysis'].append(time.time() - start_time)
        return results

    async def run_detection(self, frame: np.ndarray) -> List[Dict]:
        def sync_detect():
            if self.yolo is None: return []
            detections = []
            try:
                results = self.yolo(frame, verbose=False)
                for r in results:
                    if r.boxes:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf, cls = box.conf[0].item(), int(box.cls[0].item())
                            label = self.yolo.names[cls]
                            detections.append({'label': label, 'confidence': conf, 'bbox': [x1, y1, x2, y2]})
            except Exception as e:
                logger.error(f"Detection failed: {e}")
            return detections
        return await self._run_in_executor(sync_detect)

    async def process_video(self, video_path: str, query: str) -> Dict[str, Any]:
        pipeline_start = time.time()
        video_name = Path(video_path).stem
        output_dir = f"output_{video_name}"
        Path(output_dir).mkdir(exist_ok=True)
        logger.info(f"Processing {video_path} with query: '{query}'")

        frames = await self.extract_frames(video_path)
        if not frames:
            return {}

        analyses = await self.analyze_frames(frames, query)
        selected_analyses = sorted(analyses, key=lambda x: x.relevance_score, reverse=True)[:self.top_k]
        
        frame_dict = dict(frames)
        for analysis in selected_analyses:
            if (frame_data := frame_dict.get(analysis.timestamp)) is not None:
                detections = await self.run_detection(frame_data)
                analysis.detections = detections
                analysis.crop_paths = [
                    self._save_crop(frame_data, det, analysis.timestamp, video_path, output_dir)
                    for det in detections
                    if any(qw in det['label'].lower() for qw in query.lower().split())
                ]
                analysis.frame_path = self._save_annotated_frame(frame_data, detections, analysis.timestamp, output_dir)

        report = self._generate_report(query, selected_analyses, video_path, len(frames), output_dir)
        self.timings['total'].append(time.time() - pipeline_start)
        logger.info(f"Pipeline for {video_path} completed in {self.timings['total'][-1]:.2f}s")
        return report

    def _save_crop(self, frame, det, ts, video_path, output_dir):
        x1, y1, x2, y2 = map(int, det['bbox'])
        crop = frame[y1:y2, x1:x2]
        crop_filename = f"{Path(video_path).stem}{ts:.2f}s{det['label']}.jpg"
        crop_path = str(Path(output_dir) / crop_filename)
        cv2.imwrite(crop_path, crop)
        return crop_path

    def _save_annotated_frame(self, frame, detections, ts, output_dir):
        img = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        frame_filename = f"frame_at_{ts:.2f}s.jpg"
        frame_path = str(Path(output_dir) / frame_filename)
        cv2.imwrite(frame_path, img)
        return frame_path

    def _generate_report(self, query, analyses, video_path, total_frames, output_dir):
        answer = "No relevant content found."
        if analyses and analyses[0].relevance_score > 0.1:
            top_frame = analyses[0]
            answer = f"Most relevant content at {top_frame.timestamp:.2f}s: '{top_frame.caption}'"
        
        report = {'query': query, 'video': video_path, 'final_answer': answer, 'top_frames': [asdict(f) for f in analyses]}
        
        with open(Path(output_dir) / "report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        generate_markdown_report(report, output_dir)
        
        return report


# ===================== Concurrent Execution (callable) =====================
async def run_concurrent(video_files_map: Dict[str, str], queries: Dict[str, str]) -> List[Dict[str, Any]]:
    vlm_server = GemmaVLMServer()
    if vlm_server.model is None:
        logger.error("VLM Model could not be loaded. Aborting analysis.")
        return []

    executor = ThreadPoolExecutor(max_workers=4)

    tasks = []
    for video_path in video_files_map.keys():
        video_name = Path(video_path).stem
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 10000:
            logger.warning(f"Skipping {video_path} as it is missing or invalid.")
            continue
        if video_name in queries:
            query = queries[video_name]
            pipeline = VideoPipeline(vlm_server, executor)
            tasks.append(pipeline.process_video(video_path, query))
        else:
            logger.warning(f"No query found for {video_name}, skipping.")

    if not tasks:
        logger.error("No valid videos to process.")
        return []

    logger.info(f"Launching analysis for {len(tasks)} videos concurrently...")
    results = await asyncio.gather(*tasks)
    return results


