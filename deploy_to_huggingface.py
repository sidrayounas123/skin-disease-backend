#!/usr/bin/env python3
"""
Script to deploy the Skin Disease Detection API to Hugging Face Spaces
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and print the result"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print("SUCCESS:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED:")
        print(f"Exit code: {e.returncode}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main deployment function"""
    print("SKIN DISEASE DETECTION API - HUGGING FACE DEPLOYMENT")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists("app/main.py"):
        print("ERROR: Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Check current git status
    if not run_command("git status", "Check current git status"):
        print("ERROR: Git status check failed")
        sys.exit(1)
    
    # Step 2: Add all changes
    if not run_command("git add .", "Add all changes"):
        print("ERROR: Failed to add changes")
        sys.exit(1)
    
    # Step 3: Commit changes
    commit_msg = """Deploy to Hugging Face Spaces with skin detection fixes

- Fixed skin detector to load weights from model2.pth
- Integrated skin detection into both prediction endpoints
- Added proper error handling for model loading
- Prevents misclassification of non-skin images
- All fixes applied and tested successfully"""

    if not run_command(f'git commit -m "{commit_msg}"', "Commit changes"):
        print("INFO: No new changes to commit")
    
    # Step 4: Push to GitHub
    if not run_command("git push", "Push changes to GitHub"):
        print("ERROR: Failed to push to GitHub")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("DEPLOYMENT INSTRUCTIONS FOR HUGGING FACE")
    print("="*60)
    print("""
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in the details:
   - Space name: skin-disease-backend
   - SDK: Docker
   - Hardware: CPU Basic (or your preference)
   - Space visibility: Public or Private
4. Connect to your GitHub repository: sidrayounas123/skin-disease-backend
5. Hugging Face will automatically build and deploy

The deployment will take 5-10 minutes. Once complete, your API will be available at:
https://huggingface.co/spaces/[your-username]/skin-disease-backend

Features included in this deployment:
- Both prediction endpoints with skin detection
- Model 1 and Model 2 loading with error handling
- Firebase integration (graceful fallback if not available)
- Proper irrelevant image detection
- All recent fixes for misclassification issues
""")
    
    print("\n" + "="*60)
    print("DEPLOYMENT READY!")
    print("="*60)
    print("All changes have been pushed to GitHub.")
    print("Follow the instructions above to complete Hugging Face deployment.")

if __name__ == "__main__":
    main()
