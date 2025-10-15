#!/usr/bin/env python3
"""
Video Upload Helper
Helps organize and validate video files for analysis.
"""

import os
import shutil
from pathlib import Path
from typing import List


def get_supported_extensions() -> List[str]:
    """Get list of supported video file extensions."""
    return ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']


def validate_video_file(file_path: Path) -> bool:
    """Validate if file is a supported video format."""
    if not file_path.exists():
        return False
    
    if not file_path.is_file():
        return False
    
    # Check file extension
    if file_path.suffix.lower() not in get_supported_extensions():
        return False
    
    # Check file size (must be at least 1KB)
    if file_path.stat().st_size < 1024:
        return False
    
    return True


def upload_video_to_folder(source_path: str, videos_dir: str = "videos") -> bool:
    """Upload a video file to the videos directory."""
    source = Path(source_path)
    videos_path = Path(videos_dir)
    
    # Create videos directory if it doesn't exist
    videos_path.mkdir(exist_ok=True)
    
    # Validate source file
    if not validate_video_file(source):
        print(f"❌ Invalid video file: {source}")
        return False
    
    # Check if file already exists
    destination = videos_path / source.name
    if destination.exists():
        print(f"⚠️  File already exists: {destination}")
        overwrite = input("Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            return False
    
    try:
        # Copy file to videos directory
        shutil.copy2(source, destination)
        print(f"✅ Uploaded: {source.name} -> {destination}")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def list_videos_in_folder(videos_dir: str = "videos") -> List[Path]:
    """List all videos in the videos directory."""
    videos_path = Path(videos_dir)
    
    if not videos_path.exists():
        return []
    
    video_files = []
    for file_path in videos_path.iterdir():
        if validate_video_file(file_path):
            video_files.append(file_path)
    
    return sorted(video_files)


def main():
    print("📁 Video Upload Helper")
    print("=" * 30)
    
    while True:
        print("\nOptions:")
        print("1. Upload video file")
        print("2. List videos in folder")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            source_path = input("Enter path to video file: ").strip()
            if source_path:
                upload_video_to_folder(source_path)
        
        elif choice == '2':
            videos = list_videos_in_folder()
            if videos:
                print(f"\n📹 Videos in 'videos' folder:")
                for i, video in enumerate(videos, 1):
                    size_mb = video.stat().st_size / (1024 * 1024)
                    print(f"  {i:2d}. {video.name:<30} ({size_mb:.1f} MB)")
            else:
                print("\n📭 No videos found in 'videos' folder.")
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("❌ Invalid option. Please select 1-3.")


if __name__ == "__main__":
    main()
