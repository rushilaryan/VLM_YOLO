#!/usr/bin/env python3
"""
Runner for concurrent video analysis pipeline based on provided Colab code.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.concurrent_pipeline import run_concurrent


def download_file(url: str, dest_path: Path) -> bool:
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return dest_path.exists() and dest_path.stat().st_size > 10000
    except Exception as e:
        logging.error(f"Download failed: {url} -> {dest_path}: {e}")
        return False


async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Scan videos folder for video files
    videos_dir = Path("videos")
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    
    video_files = []
    if videos_dir.exists():
        for file_path in videos_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(file_path)
    
    # Fallback to test_video.mp4 in root if videos folder is empty
    if not video_files:
        test_video_path = Path("test_video.mp4")
        if test_video_path.exists():
            video_files = [test_video_path]
        else:
            logging.error("No video files found in 'videos' folder or 'test_video.mp4' in root directory.")
            logging.info("Please add video files to the 'videos' folder or place 'test_video.mp4' in the project root.")
            return
    
    # Create video files mapping
    video_files_map = {str(video): str(video) for video in video_files}
    
    # Create queries for each video
    queries = {}
    for video_file in video_files:
        video_stem = video_file.stem
        queries[video_stem] = f"Analyze the content of {video_stem} video"

    results = await run_concurrent(video_files_map, queries)

    for report in results:
        if report:
            print("\n" + "="*50)
            print(f"✅ ANALYSIS COMPLETE FOR: {report['video']}")
            print("="*50)
            print(f"Final Answer: {report['final_answer']}")
            print(f"Full reports saved to output_{Path(report['video']).stem}/")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass



