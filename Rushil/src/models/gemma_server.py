#!/usr/bin/env python3
"""
Gemma Vision-Language Model Server
Serves the Gemma 3 VLM locally with FastAPI
"""

import os
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import base64
import io

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger(__name__)

@dataclass
class FrameAnalysis:
    """Analysis result for a single frame"""
    caption: str
    entities: List[str]
    reasoning: str
    relevance_score: float

class AnalyzeRequest(BaseModel):
    """Request model for image analysis"""
    images: List[str]  # base64 encoded images
    query: str

class AnalyzeResponse(BaseModel):
    """Response model for image analysis"""
    results: List[Dict[str, Any]]
    processing_time: float
    batch_size: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    latency: float
    gpu_available: bool
    gpu_utilization: Optional[float]
    model_loaded: bool

class GemmaVLMServer:
    """Gemma Vision-Language Model Server"""
    
    def __init__(self, model_name: str = "google/paligemma-3b-pt-224", device: str = None):
        """Initialize Gemma VLM"""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        logger.info(f"Initializing Gemma VLM on {self.device}")
        # Load the real Gemma VLM model
        self._load_model()
    
    def _load_model(self):
        """Load the Gemma model and processor"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
                low_cpu_mem_usage=True
            )
            
            if self.device != 'cuda':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
            logger.warning("Running in mock mode - no actual model loaded")
    
    def analyze_image_batch(self, images: List[Image.Image], query: str) -> List[Dict[str, Any]]:
        """Analyze a batch of images with a single query"""
        start_time = time.time()
        
        # Only use real Gemma model - no mock mode
        if not self.model_loaded or self.model is None:
            logger.error("Gemma model not loaded! Please check authentication and model loading.")
            return [{
                'caption': 'Model not available - please authenticate with HuggingFace',
                'entities': [],
                'reasoning': 'Gemma model failed to load',
                'relevance_score': 0.0
            } for _ in images]

        try:
            # Create prompts for each image
            model_prompt = f"<image>\nBased on the query '{query}', describe what you see in this image. Focus on elements that are relevant to the query."
            prompts = [model_prompt] * len(images)
            
            # Process inputs
            inputs = self.processor(
                images=images, 
                text=prompts, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Generate responses
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode responses
            responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
            # Parse results
            results = []
            for response in responses:
                clean_response = response.replace(model_prompt, "").strip()
                parsed_result = self._parse_response(clean_response, query)
                results.append(parsed_result)
                
            processing_time = time.time() - start_time
            logger.info(f"Analyzed {len(images)} images in {processing_time:.2f}s")
            
            return results

        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return [{
                'caption': f'Analysis failed: {str(e)}',
                'entities': [],
                'reasoning': f'Error during processing: {str(e)}',
                'relevance_score': 0.0
            } for _ in images]

    def _parse_response(self, response: str, query: str) -> Dict[str, Any]:
        """Parse model response into structured format"""
        # Calculate relevance score based on query-word overlap
        response_words = set(response.lower().split())
        query_words = set(query.lower().split())
        common_words = response_words.intersection(query_words)
        
        # Jaccard-like similarity
        relevance = len(common_words) / (len(query_words) + 1e-6) if query_words else 0.0
        
        # Extract entities (words longer than 3 chars, not in query)
        entities = [word for word in response.split() 
                   if len(word) > 3 and word.lower() not in query_words][:5]
        
        return {
            'caption': response[:300],  # Limit caption length
            'entities': entities,
            'reasoning': f"Found {len(common_words)} matching keywords with query",
            'relevance_score': min(relevance, 1.0)  # Cap at 1.0
        }
    
    def get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization if available"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization(0) if torch.cuda.device_count() > 0 else None
        except:
            pass
        return None

# Global server instance
gemma_server = GemmaVLMServer()
app = FastAPI(title="Gemma VLM Server", version="1.0.0")

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_images(request: AnalyzeRequest):
    """Analyze images with the given query"""
    try:
        start_time = time.time()
        
        # Decode base64 images
        images = []
        for img_b64 in request.images:
            try:
                img_data = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_data))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                logger.error(f"Failed to decode image: {e}")
                continue
        
        if not images:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        # Analyze images
        results = gemma_server.analyze_image_batch(images, request.query)
        
        processing_time = time.time() - start_time
        
        return AnalyzeResponse(
            results=results,
            processing_time=processing_time,
            batch_size=len(images)
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    start_time = time.time()
    
    # Simulate some work for latency measurement
    await asyncio.sleep(0.001)
    
    latency = time.time() - start_time
    gpu_util = gemma_server.get_gpu_utilization()
    
    return HealthResponse(
        status="healthy" if gemma_server.model_loaded else "degraded",
        latency=latency,
        gpu_available=torch.cuda.is_available(),
        gpu_utilization=gpu_util,
        model_loaded=gemma_server.model_loaded
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Analysis Pipeline - VLM Server",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze images with query",
            "/health": "GET - Health check",
            "/metrics": "GET - System metrics"
        }
    }

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring"""
    import psutil
    
    metrics_data = {
        "model_loaded": gemma_server.model_loaded,
        "gpu_available": torch.cuda.is_available(),
        "gpu_utilization": gemma_server.get_gpu_utilization(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    
    if torch.cuda.is_available():
        metrics_data.update({
            "gpu_memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,  # GB
            "gpu_memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,    # GB
        })
    
    return metrics_data

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    logger.info(f"Starting Gemma VLM server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
