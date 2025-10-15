#!/usr/bin/env python3
"""
Start the Gemma VLM server
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gemma_server import run_server

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Gemma VLM Server...")
    
    run_server(host="0.0.0.0", port=8000)
