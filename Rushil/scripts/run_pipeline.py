#!/usr/bin/env python3
"""
Run the video analysis pipeline
"""

import sys
import os
import asyncio
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.video_processor import VideoProcessor

async def main():
    parser = argparse.ArgumentParser(description="Video Analysis Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--server-url", default="http://localhost:8000", help="VLM server URL")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction rate")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top frames to select")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for LLM analysis")
    parser.add_argument("--yolo-model", default="yolov8x.pt", help="YOLO model to use")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return 1

    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    logger.info("Starting Video Analysis Pipeline")
    logger.info(f"Video: {args.video}")
    logger.info(f"Query: {args.query}")
    logger.info(f"Output: {args.output}")
  
    processor = VideoProcessor(
        vlm_server_url=args.server_url,
        fps_rate=args.fps,
        top_k=args.top_k,
        batch_size=args.batch_size,
        yolo_model=args.yolo_model
    )
    
    try:
        # Process video
        report = await processor.process_video(args.video, args.query, args.output)
        
        if report:
            logger.info("Pipeline completed successfully!")
            logger.info(f"Report saved to: {output_path}/report.json")
            logger.info(f"Markdown report saved to: {output_path}/report.md")
            
            # Print summary
            print("\n" + "="*50)
            print("ANALYSIS COMPLETE")
            print("="*50)
            print(f"Query: {report['query']}")
            print(f"Final Answer: {report['final_answer']}")
            print(f"\nPerformance:")
            perf = report['performance']
            print(f"- Total frames extracted: {perf['total_frames_extracted']}")
            print(f"- Processing FPS: {perf['processing_fps']:.2f}")
            print(f"- Total time: {perf['total_pipeline_time_s']:.2f}s")
            
            return 0
        else:
            logger.error("Pipeline failed to produce a report")
            return 1
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
