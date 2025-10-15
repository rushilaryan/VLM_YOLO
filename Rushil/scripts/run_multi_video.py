#!/usr/bin/env python3
"""
Multi-Video Analysis Pipeline
Scans videos folder and processes multiple videos with customizable queries.
"""

import os
import sys
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.concurrent_pipeline import run_concurrent


def get_video_files(videos_dir: str = "videos") -> List[Path]:
    """Scan videos directory for supported video files."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    videos_path = Path(videos_dir)
    
    if not videos_path.exists():
        logging.warning(f"Videos directory '{videos_dir}' not found. Creating it...")
        videos_path.mkdir(exist_ok=True)
        return []
    
    video_files = []
    for file_path in videos_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    return sorted(video_files)


def load_queries_config(config_path: str = "video_queries.json") -> Dict[str, str]:
    """Load video queries from configuration file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Create default configuration
        default_queries = {
            "test_video": "Find all cats in the video",
            "sample_video": "Describe what's happening in the scene",
            "activity_video": "Find people and their activities"
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_queries, f, indent=2)
        
        logging.info(f"Created default queries config at {config_file}")
        return default_queries
    
    with open(config_file, 'r') as f:
        return json.load(f)


def interactive_video_selection(video_files: List[Path]) -> List[Path]:
    """Interactive video selection interface."""
    if not video_files:
        logging.error("No video files found in videos directory.")
        return []
    
    print("\n" + "="*60)
    print("🎬 AVAILABLE VIDEOS")
    print("="*60)
    
    for i, video_file in enumerate(video_files, 1):
        size_mb = video_file.stat().st_size / (1024 * 1024)
        print(f"{i:2d}. {video_file.name:<30} ({size_mb:.1f} MB)")
    
    print("\nOptions:")
    print("  - Enter numbers (e.g., 1,3,5) to select specific videos")
    print("  - Enter 'all' to process all videos")
    print("  - Enter 'skip' to exit")
    
    while True:
        choice = input("\nYour selection: ").strip().lower()
        
        if choice == 'skip':
            return []
        elif choice == 'all':
            return video_files
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                selected = []
                for idx in indices:
                    if 1 <= idx <= len(video_files):
                        selected.append(video_files[idx - 1])
                    else:
                        print(f"Invalid selection: {idx}")
                
                if selected:
                    return selected
                else:
                    print("No valid videos selected. Please try again.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas, 'all', or 'skip'.")


def create_video_queries(video_files: List[Path], queries_config: Dict[str, str]) -> Dict[str, str]:
    """Create video queries mapping."""
    video_queries = {}
    
    for video_file in video_files:
        video_stem = video_file.stem
        
        if video_stem in queries_config:
            video_queries[video_file.name] = queries_config[video_stem]
        else:
            # Default query for unknown videos
            default_query = f"Analyze the content of {video_stem} video"
            video_queries[video_file.name] = default_query
            logging.info(f"Using default query for {video_file.name}")
    
    return video_queries


async def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('multi_video_analysis.log')
        ]
    )
    
    print("🎬 Multi-Video Analysis Pipeline")
    print("=" * 50)
    
    # Scan for video files
    video_files = get_video_files()
    
    if not video_files:
        print("\n❌ No video files found in 'videos' directory.")
        print("Please add video files (.mp4, .avi, .mov, etc.) to the videos folder.")
        return
    
    # Load queries configuration
    queries_config = load_queries_config()
    
    # Interactive video selection
    selected_videos = interactive_video_selection(video_files)
    
    if not selected_videos:
        print("No videos selected. Exiting.")
        return
    
    # Create video files mapping and queries
    video_files_map = {str(video): str(video) for video in selected_videos}
    video_queries = create_video_queries(selected_videos, queries_config)
    
    print(f"\n🚀 Processing {len(selected_videos)} videos:")
    for video in selected_videos:
        print(f"  - {video.name}")
    
    print(f"\n📝 Queries:")
    for video_name, query in video_queries.items():
        print(f"  - {video_name}: {query}")
    
    # Confirm processing
    confirm = input("\nProceed with analysis? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Analysis cancelled.")
        return
    
    # Run concurrent analysis
    print("\n🔄 Starting concurrent analysis...")
    results = await run_concurrent(video_files_map, video_queries)
    
    # Display results
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS")
    print("="*60)
    
    for i, report in enumerate(results, 1):
        if report:
            print(f"\n{i}. ✅ {Path(report['video']).name}")
            print(f"   Query: {report['query']}")
            print(f"   Answer: {report['final_answer']}")
            print(f"   Reports: output_{Path(report['video']).stem}/")
    
    print(f"\n🎉 Analysis complete! Check individual output directories for detailed reports.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        print(f"\n❌ Error: {e}")
