#!/usr/bin/env python3
"""
Test suite for the video analysis pipeline
"""

import sys
import os
import json
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.video_processor import VideoProcessor

async def run_test_case(test_case, processor, results):
    """Run a single test case"""
    video_path = Path(__file__).parent / test_case['video']
    
    if not video_path.exists():
        results[test_case['video']] = {
            'status': 'skipped',
            'reason': 'Video file not found'
        }
        return
    
    try:
        output_dir = f"test_output_{test_case['video'].replace('.mp4', '')}"
        report = await processor.process_video(str(video_path), test_case['query'], output_dir)
        
        if report:
            # Evaluate results
            relevance_scores = [frame['relevance_score'] for frame in report['top_frames']]
            max_relevance = max(relevance_scores) if relevance_scores else 0
            
            # Check detected objects
            all_detections = []
            for frame in report['top_frames']:
                if frame['detections']:
                    all_detections.extend([det['label'] for det in frame['detections']])
            
            unique_detections = list(set(all_detections))
            
            # Evaluate against rubric
            rubric = test_case.get('rubric', {})
            passed = (
                max_relevance >= rubric.get('relevance_score_threshold', 0.3) and
                len(unique_detections) >= rubric.get('min_objects_detected', 1)
            )
            
            results[test_case['video']] = {
                'status': 'passed' if passed else 'failed',
                'max_relevance_score': max_relevance,
                'detected_objects': unique_detections,
                'processing_time': report['performance']['total_pipeline_time_s'],
                'frames_analyzed': len(report['top_frames']),
                'final_answer': report['final_answer']
            }
        else:
            results[test_case['video']] = {
                'status': 'failed',
                'reason': 'No report generated'
            }
            
    except Exception as e:
        results[test_case['video']] = {
            'status': 'error',
            'error': str(e)
        }

async def main():
    """Run all test cases"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load test cases
    test_file = Path(__file__).parent / "test_queries.json"
    with open(test_file) as f:
        test_data = json.load(f)
    
    test_cases = test_data['test_cases']
    rubric = test_data['rubric']
    
    logger.info(f"Running {len(test_cases)} test cases")
    
    # Initialize processor
    processor = VideoProcessor(
        vlm_server_url="http://localhost:8000",
        fps_rate=1.0,
        top_k=5,
        batch_size=4  # Smaller batch for testing
    )
    
    # Run tests
    results = {}
    for test_case in test_cases:
        logger.info(f"Running test: {test_case['video']} - {test_case['description']}")
        await run_test_case(test_case, processor, results)
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for video, result in results.items():
        status = result['status']
        print(f"\n{video}: {status.upper()}")
        
        if status == 'passed':
            passed += 1
            print(f"  Relevance Score: {result['max_relevance_score']:.3f}")
            print(f"  Objects Detected: {result['detected_objects']}")
            print(f"  Processing Time: {result['processing_time']:.2f}s")
        elif status == 'failed':
            failed += 1
            if 'reason' in result:
                print(f"  Reason: {result['reason']}")
            else:
                print(f"  Relevance Score: {result['max_relevance_score']:.3f}")
                print(f"  Objects Detected: {result['detected_objects']}")
        elif status == 'skipped':
            skipped += 1
            print(f"  Reason: {result['reason']}")
        else:  # error
            failed += 1
            print(f"  Error: {result['error']}")
    
    print(f"\nSUMMARY:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(results)}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
