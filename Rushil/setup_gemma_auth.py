#!/usr/bin/env python3
"""
Setup script for Gemma model authentication
"""

import os
from huggingface_hub import login

def setup_gemma_auth():
    """Setup HuggingFace authentication for Gemma model"""
    print("=== Gemma VLM Authentication Setup ===")
    print()
    print("To use the real Gemma model, you need a HuggingFace token.")
    print("Follow these steps:")
    print()
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'Read' permissions")
    print("3. Copy the token")
    print()
    
    token = input("Enter your HuggingFace token: ").strip()
    
    if not token:
        print("No token provided. Exiting.")
        return False
    
    try:
        # Login with the token
        login(token=token)
        print("✅ Authentication successful!")
        print("You can now run the Gemma VLM server.")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

if __name__ == "__main__":
    setup_gemma_auth()




