#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HeyGen 4-Problem Video Generator v4 - Enhanced with Success Principles

This script processes chunked 4-problem video scripts and generates videos using HeyGen Template API:
- Reads chunked scripts from four-problem-script-drafts bucket
- Parses {{four_prob_slide_XX_content}} placeholders (15 slides) - FIXED NAMING
- Calls HeyGen Template API using proven success pattern from 4-slide curl
- Monitors video generation status and uploads completed videos to bucket
- Enhanced error handling, rate limiting, and test mode

SUCCESS PRINCIPLES APPLIED:
- Template API endpoint: /v2/template/{template_id}/generate 
- Correct variable structure: {"name": "...", "type": "text", "properties": {"content": "..."}}
- Payload structure: {"caption": false, "title": "...", "variables": {...}}
- Placeholder naming: four_prob_slide_01_content (no leading numbers)
- One slide per scene principle
"""

# =====================================================================
# COMMON IMPORTS AND SETUP
# =====================================================================

import os
import json
import re
import time
import traceback
import logging
import requests
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

# Supabase integration
from dotenv import load_dotenv
from supabase import create_client, Client

# =====================================================================
# CONSTANTS AND CONFIGURATION
# =====================================================================

# Script directory for use in file path construction
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to environment variables file
ENV_FILE_PATH = "STATIC_VARS_MAR2025.env"

# Load environment variables
load_dotenv(ENV_FILE_PATH)
supabase_url = os.getenv("VITE_SUPABASE_URL")
supabase_service_key = os.getenv("VITE_SUPABASE_SERVICE_ROLE_KEY")
heygen_api_key = os.getenv("HEYGEN_API_KEY")

# Handle missing environment variables
if not supabase_url or not supabase_service_key:
    print("Error: Missing Supabase credentials")
    exit(1)

if not heygen_api_key:
    print("Error: Missing HeyGen API key - Please add HEYGEN_API_KEY to STATIC_VARS_MAR2025.env")
    exit(1)

supabase: Client = create_client(supabase_url, supabase_service_key)

# Record start time for execution tracking (Eastern Time)
eastern_tz = ZoneInfo("America/New_York")
start_time = datetime.now(eastern_tz)

# 4-Problem Template Configuration (15 slides) - UPDATED FOR NEW NAMING
TEMPLATE_4PROB_CONFIG = {
    "template_id": "7cc005fc62dc4db8a02a008723e67297",  # FOUR SLIDES API 27JUL25
    "slide_count": 15,
    "bucket_name": "four-problem-script-drafts",
    "script_type": "4prob",
    "output_bucket": "heygen-generated-videos",
    "structure": {
        "intro": {"slides": [1], "count": 1},
        "problem_1": {"slides": [2, 3, 4], "count": 3},
        "problem_2": {"slides": [5, 6, 7], "count": 3},
        "problem_3": {"slides": [8, 9, 10], "count": 3},
        "problem_4": {"slides": [11, 12, 13], "count": 3},
        "outro_cta": {"slides": [14, 15], "count": 2}
    }
}

# Enhanced retry settings
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 3  # seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [HEYGEN-4PROB-VIDEO] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"heygen_4prob_video_log_{datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}.log")
    ])
logger = logging.getLogger(__name__)

# Log initialization for HeyGen 4-problem video generation
logger.info("=" * 60)
logger.info("HEYGEN 4-PROBLEM VIDEO GENERATOR v4 - SESSION START")
logger.info(f"Target: Generate videos from chunked 4-problem scripts (15 slides)")
logger.info(f"Template ID: {TEMPLATE_4PROB_CONFIG['template_id']}")
logger.info(f"Processing bucket: {TEMPLATE_4PROB_CONFIG['bucket_name']}")
logger.info(f"Output bucket: {TEMPLATE_4PROB_CONFIG['output_bucket']}")
logger.info(f"Session ID: {datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}")
logger.info("=" * 60)

# Suppress excessive HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Setup directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_PATH = os.path.join(script_dir, "STATIC_VARS_MAR2025.env")

# =====================================================================
# COMMON FUNCTIONS (COPIED FROM EXISTING SCRIPTS)
# =====================================================================

def fetch_configuration_from_supabase(config_name: Optional[str] = None, config_id: Optional[int] = None) -> Dict:
    """
    Fetch configuration variables from Supabase workflow_configs table
    """
    try:
        logger.info(f"[HEYGEN-4PROB-VIDEO-CONFIG] Fetching configuration from Supabase...")

        if config_id:
            logger.info(f"[HEYGEN-4PROB-VIDEO-CONFIG] Fetching configuration with ID: {config_id}")
            response = supabase.table("workflow_configs").select("*").eq("id", config_id).execute()
        elif config_name:
            logger.info(f"[HEYGEN-4PROB-VIDEO-CONFIG] Fetching configuration with name: {config_name}")
            response = supabase.table("workflow_configs").select("*").eq("config_name", config_name).execute()
        else:
            logger.info("[HEYGEN-4PROB-VIDEO-CONFIG] Fetching most recent configuration")
            response = supabase.table("workflow_configs").select("*").order("created_at", desc=True).limit(1).execute()

        if not response.data or len(response.data) == 0:
            raise Exception("No configuration found in Supabase")

        config_data = response.data[0]
        config_name = config_data.get('config_name', 'unnamed')
        logger.info(f"[HEYGEN-4PROB-VIDEO-CONFIG] Successfully fetched configuration: {config_name}")
        return config_data.get("variables", {})

    except Exception as e:
        logger.error(f"[HEYGEN-4PROB-VIDEO-CONFIG] Error fetching configuration from Supabase: {str(e)}")
        raise


def download_file_from_bucket(bucket_name: str, file_name: str) -> str:
    """
    Download a file from Supabase storage bucket
    """
    try:
        logger.info(f"[HEYGEN-4PROB-VIDEO-DOWNLOAD] Downloading {file_name} from {bucket_name} bucket...")
        response = supabase.storage.from_(bucket_name).download(file_name)

        if response:
            content = response.decode('utf-8')
            logger.info(f"[HEYGEN-4PROB-VIDEO-DOWNLOAD] Successfully downloaded {file_name} ({len(content)} characters)")
            return content
        else:
            raise Exception(f"Failed to download {file_name}")

    except Exception as e:
        logger.error(f"[HEYGEN-4PROB-VIDEO-DOWNLOAD] Error downloading {file_name} from {bucket_name}: {str(e)}")
        raise


def upload_file_to_bucket(bucket_name: str, file_name: str, file_content: str) -> bool:
    """
    Upload a file to Supabase storage bucket
    """
    try:
        logger.info(f"[HEYGEN-4PROB-VIDEO-UPLOAD] Uploading {file_name} to {bucket_name} bucket...")

        # Convert string content to bytes
        file_bytes = file_content.encode('utf-8')

        response = supabase.storage.from_(bucket_name).upload(
            file_name, file_bytes, {"content-type": "text/plain"})

        logger.info(f"[HEYGEN-4PROB-VIDEO-UPLOAD] Successfully uploaded {file_name} to {bucket_name}")
        return True

    except Exception as e:
        logger.error(f"[HEYGEN-4PROB-VIDEO-UPLOAD] Error uploading {file_name} to {bucket_name}: {str(e)}")
        raise


def list_files_in_bucket(bucket_name: str) -> List[str]:
    """
    List all files in a Supabase storage bucket
    """
    try:
        logger.info(f"[HEYGEN-4PROB-VIDEO-LIST] Listing files in {bucket_name} bucket...")
        response = supabase.storage.from_(bucket_name).list()

        if response:
            file_names = [file['name'] for file in response if file['name'].endswith('.txt')]
            logger.info(f"[HEYGEN-4PROB-VIDEO-LIST] Found {len(file_names)} .txt files in {bucket_name}")
            return file_names
        else:
            return []

    except Exception as e:
        logger.error(f"[HEYGEN-4PROB-VIDEO-LIST] Error listing files in {bucket_name}: {str(e)}")
        raise

# =====================================================================
# ENHANCED VIDEO UPLOAD FUNCTIONS
# =====================================================================

def upload_video_to_bucket(video_url: str, output_filename: str, bucket_name: str = "heygen-generated-videos") -> Tuple[bool, str]:
    """
    Download video from HeyGen and upload directly to Supabase bucket
    
    Args:
        video_url: Direct URL to video file from HeyGen
        output_filename: Filename for the video in bucket
        bucket_name: Supabase storage bucket name
        
    Returns:
        Tuple of (success: bool, bucket_url: str)
    """
    try:
        logger.info(f"[HEYGEN-4PROB-VIDEO-UPLOAD] Downloading and uploading video: {output_filename}")
        
        # Download video with streaming to handle large files
        response = requests.get(video_url, timeout=600, stream=True)  # 10 min timeout
        
        if response.status_code == 200:
            # Get video content
            video_content = response.content
            file_size = len(video_content)
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"[HEYGEN-4PROB-VIDEO-UPLOAD] Downloaded video: {file_size_mb:.1f} MB")
            
            # Upload directly to bucket
            upload_response = supabase.storage.from_(bucket_name).upload(
                output_filename, 
                video_content, 
                {"content-type": "video/mp4", "cache-control": "3600"}
            )
            
            # Generate public URL for the uploaded video
            bucket_url = f"{supabase_url}/storage/v1/object/public/{bucket_name}/{output_filename}"
            
            logger.info(f"[HEYGEN-4PROB-VIDEO-UPLOAD] Successfully uploaded video: {output_filename} ({file_size_mb:.1f} MB)")
            logger.info(f"[HEYGEN-4PROB-VIDEO-UPLOAD] Bucket URL: {bucket_url}")
            return True, bucket_url
        else:
            raise Exception(f"Failed to download video: {response.status_code}")
            
    except Exception as e:
        logger.error(f"[HEYGEN-4PROB-VIDEO-UPLOAD] Error uploading video: {str(e)}")
        return False, ""

# =====================================================================
# ENHANCED HEYGEN API FUNCTIONS - APPLYING SUCCESS PRINCIPLES
# =====================================================================

def enhanced_rate_limit_handler(attempt: int, error_msg: str, retry_delay: int) -> int:
    """
    Enhanced rate limiting with progressive backoff and specific HeyGen handling
    
    Returns: Updated retry delay in seconds
    """
    error_lower = error_msg.lower()
    
    if "rate" in error_lower and "limit" in error_lower:
        # HeyGen rate limit hit
        if "minute" in error_lower:
            # Per-minute limit - wait longer
            backoff_delay = 70 + (attempt * 30)  # 70s, 100s, 130s, etc.
            logger.warning(f"[HEYGEN-4PROB-VIDEO-RATE-LIMIT] Per-minute rate limit - waiting {backoff_delay}s")
        elif "hour" in error_lower:
            # Hourly limit - much longer wait
            backoff_delay = 3600 + (attempt * 600)  # 1hr+, 1hr 10min+, etc.
            logger.warning(f"[HEYGEN-4PROB-VIDEO-RATE-LIMIT] Hourly rate limit - waiting {backoff_delay}s")
        else:
            # Generic rate limit
            backoff_delay = retry_delay * 3
            logger.warning(f"[HEYGEN-4PROB-VIDEO-RATE-LIMIT] Generic rate limit - waiting {backoff_delay}s")
        return backoff_delay
    
    elif "quota" in error_lower or "billing" in error_lower:
        # Account/billing issue - longer wait
        backoff_delay = 300 + (attempt * 300)  # 5min, 10min, 15min, etc.
        logger.warning(f"[HEYGEN-4PROB-VIDEO-QUOTA] Quota/billing issue - waiting {backoff_delay}s")
        return backoff_delay
    
    elif "timeout" in error_lower or "connection" in error_lower:
        # Network issues - moderate backoff
        backoff_delay = retry_delay * 2
        logger.warning(f"[HEYGEN-4PROB-VIDEO-NETWORK] Network issue - waiting {backoff_delay}s")
        return backoff_delay
    
    else:
        # Standard exponential backoff
        return retry_delay * 2


def parse_chunked_script_placeholders(script_content: str) -> Dict[str, str]:
    """
    Parse chunked script content to extract placeholder mappings - UPDATED FOR NEW NAMING
    
    Args:
        script_content: Content of chunked script file
        
    Returns:
        Dictionary mapping slide names to content
    """
    try:
        logger.info("[HEYGEN-4PROB-VIDEO-PARSE] Parsing chunked script placeholders...")
        
        # FIXED: Pattern to match new naming format: {{four_prob_slide_XX_content}}
        placeholder_pattern = r'\{\{(four_prob_slide_\d{2}_content)\}\}\s*\n(.+?)(?=\n\{\{four_prob_slide_\d{2}_content\}\}|\n#|\Z)'
        
        matches = re.findall(placeholder_pattern, script_content, re.DOTALL)
        
        slide_content = {}
        for placeholder_name, content in matches:
            clean_content = content.strip()
            if clean_content:
                slide_content[placeholder_name] = clean_content
                logger.debug(f"[HEYGEN-4PROB-VIDEO-PARSE] {placeholder_name}: {len(clean_content)} chars")
        
        logger.info(f"[HEYGEN-4PROB-VIDEO-PARSE] Parsed {len(slide_content)} slide placeholders")
        
        # Validate we have expected slides (1-15)
        expected_slides = [f"four_prob_slide_{i:02d}_content" for i in range(1, 16)]
        missing_slides = [slide for slide in expected_slides if slide not in slide_content]
        if missing_slides:
            logger.warning(f"[HEYGEN-4PROB-VIDEO-PARSE] Missing slides: {missing_slides}")
        
        return slide_content
        
    except Exception as e:
        logger.error(f"[HEYGEN-4PROB-VIDEO-PARSE] Error parsing placeholders: {str(e)}")
        raise


def generate_video_with_retry(template_id: str, slide_content: Dict[str, str], video_title: str, 
                             max_retries: int = DEFAULT_MAX_RETRIES, retry_delay: int = DEFAULT_RETRY_DELAY) -> str:
    """
    Generate video using HeyGen Template API with SUCCESS PRINCIPLES APPLIED
    
    Args:
        template_id: HeyGen template ID
        slide_content: Dictionary of slide content mapped to placeholder names
        video_title: Title for the generated video
        max_retries: Maximum number of retry attempts
        retry_delay: Initial retry delay in seconds
        
    Returns:
        Video ID from HeyGen API
    """
    # SUCCESS PRINCIPLE: Use exact header format from successful curl
    headers = {
        'X-Api-Key': heygen_api_key,  # Exact format from success
        'Content-Type': 'application/json'
    }
    
    # SUCCESS PRINCIPLE: Build variables payload exactly like successful 4-slide curl
    variables = {}
    for placeholder_name, content in slide_content.items():
        variables[placeholder_name] = {
            "name": placeholder_name,
            "type": "text",
            "properties": {
                "content": content
            }
        }
    
    # SUCCESS PRINCIPLE: Use exact payload structure from successful curl
    payload = {
        "caption": False,
        "title": video_title,
        "variables": variables
    }
    
    # SUCCESS PRINCIPLE: Use Template API endpoint (not video/generate)
    url = f'https://api.heygen.com/v2/template/{template_id}/generate'
    
    logger.info(f"[HEYGEN-4PROB-VIDEO-GENERATE] Using template endpoint: {url}")
    logger.info(f"[HEYGEN-4PROB-VIDEO-GENERATE] Payload structure: caption={payload['caption']}, title='{payload['title']}', variables={len(variables)} slides")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"[HEYGEN-4PROB-VIDEO-GENERATE] Attempt {attempt + 1}/{max_retries} - Generating video: {video_title}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            logger.info(f"[HEYGEN-4PROB-VIDEO-GENERATE] Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"[HEYGEN-4PROB-VIDEO-GENERATE] Response data: {data}")
                
                if 'data' in data and 'video_id' in data['data']:
                    video_id = data['data']['video_id']
                    logger.info(f"[HEYGEN-4PROB-VIDEO-GENERATE] SUCCESS - Video ID: {video_id}")
                    return video_id
                else:
                    raise Exception(f"Unexpected response format: {data}")
            elif response.status_code == 401:
                raise Exception("HeyGen API authentication failed - check your API key")
            elif response.status_code == 403:
                raise Exception("HeyGen API access forbidden - check your subscription")
            else:
                error_text = response.text
                logger.error(f"[HEYGEN-4PROB-VIDEO-GENERATE] API Error Response: {error_text}")
                raise Exception(f"HeyGen API request failed: {response.status_code} - {error_text}")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[HEYGEN-4PROB-VIDEO-ERROR] Attempt {attempt + 1} failed: {error_msg}")
            
            if attempt < max_retries - 1:
                # Use enhanced rate limit handler
                retry_delay = enhanced_rate_limit_handler(attempt, error_msg, retry_delay)
                logger.info(f"[HEYGEN-4PROB-VIDEO-RETRY] Retrying video generation in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error(f"[HEYGEN-4PROB-VIDEO-FAILURE] Failed to generate video after {max_retries} attempts")
                raise


def check_video_status_with_retry(video_id: str, max_retries: int = DEFAULT_MAX_RETRIES, 
                                 retry_delay: int = DEFAULT_RETRY_DELAY) -> Dict:
    """
    Check video generation status with enhanced retry logic
    
    Args:
        video_id: HeyGen video ID
        max_retries: Maximum number of retry attempts
        retry_delay: Initial retry delay in seconds
        
    Returns:
        Video status data from HeyGen API
    """
    headers = {
        'Accept': 'application/json',
        'X-Api-Key': heygen_api_key
    }
    
    url = f'https://api.heygen.com/v1/video_status.get?video_id={video_id}'
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"[HEYGEN-4PROB-VIDEO-STATUS] Checking status for video: {video_id}")
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    status = data['data'].get('status', 'unknown')
                    logger.debug(f"[HEYGEN-4PROB-VIDEO-STATUS] Video {video_id} status: {status}")
                    return data['data']
                else:
                    raise Exception(f"Unexpected status response format: {data}")
            else:
                raise Exception(f"Status check failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[HEYGEN-4PROB-VIDEO-STATUS-ERROR] Attempt {attempt + 1} failed: {error_msg}")
            
            if attempt < max_retries - 1:
                retry_delay = enhanced_rate_limit_handler(attempt, error_msg, retry_delay)
                logger.info(f"[HEYGEN-4PROB-VIDEO-STATUS-RETRY] Retrying status check in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error(f"[HEYGEN-4PROB-VIDEO-STATUS-FAILURE] Failed to check status after {max_retries} attempts")
                raise


def monitor_video_generation(video_id: str, max_wait_minutes: int = 45) -> Dict:
    """
    Monitor video generation until completion or timeout
    
    Args:
        video_id: HeyGen video ID
        max_wait_minutes: Maximum time to wait for completion
        
    Returns:
        Final video status data
    """
    logger.info(f"[HEYGEN-4PROB-VIDEO-MONITOR] Monitoring video generation: {video_id}")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    check_interval = 30  # Check every 30 seconds
    
    while True:
        try:
            status_data = check_video_status_with_retry(video_id)
            status = status_data.get('status', 'unknown')
            
            if status == 'completed':
                logger.info(f"[HEYGEN-4PROB-VIDEO-MONITOR] Video generation completed: {video_id}")
                return status_data
            elif status == 'failed':
                error_msg = status_data.get('error', 'Unknown error')
                raise Exception(f"Video generation failed: {error_msg}")
            elif status in ['processing', 'pending']:
                elapsed_time = time.time() - start_time
                elapsed_minutes = elapsed_time / 60
                
                if elapsed_time > max_wait_seconds:
                    raise Exception(f"Video generation timeout after {max_wait_minutes} minutes")
                
                logger.info(f"[HEYGEN-4PROB-VIDEO-MONITOR] Video {video_id} still {status}... ({elapsed_minutes:.1f}min elapsed)")
                time.sleep(check_interval)
            else:
                logger.warning(f"[HEYGEN-4PROB-VIDEO-MONITOR] Unknown status '{status}' for video {video_id}")
                time.sleep(check_interval)
                
        except Exception as e:
            logger.error(f"[HEYGEN-4PROB-VIDEO-MONITOR] Error monitoring video {video_id}: {str(e)}")
            raise

# =====================================================================
# ENHANCED MAIN WORKFLOW FUNCTIONS
# =====================================================================

def process_single_chunked_script(filename: str, template_config: Dict) -> Dict:
    """
    Process one chunked 4-problem script file to generate video with bucket upload
    
    Args:
        filename: Name of chunked script file
        template_config: Template configuration dictionary
        
    Returns:
        Processing result dictionary
    """
    try:
        logger.info(f"[HEYGEN-4PROB-VIDEO-PROCESS] Processing {filename}")
        
        bucket_name = template_config["bucket_name"]
        template_id = template_config["template_id"]
        output_bucket = template_config.get("output_bucket", "heygen-generated-videos")
        
        # Download chunked script content
        script_content = download_file_from_bucket(bucket_name, filename)
        
        # Parse slide placeholders using new naming format
        slide_content = parse_chunked_script_placeholders(script_content)
        
        if not slide_content:
            raise Exception("No valid slide placeholders found in script")
        
        logger.info(f"[HEYGEN-4PROB-VIDEO-PROCESS] Found {len(slide_content)} slides to process")
        
        # Generate video title from filename
        base_name = filename.replace('.txt', '').replace('chunked_4prob_script_', '')
        video_title = f"4-Problem Video - {base_name}"
        
        # Generate video using HeyGen API with success principles
        video_id = generate_video_with_retry(template_id, slide_content, video_title)
        
        # Monitor video generation
        final_status = monitor_video_generation(video_id)
        
        # Extract video URL
        video_url = final_status.get('video_url')
        if not video_url:
            raise Exception("No video URL in completed status")
        
        # Generate output filename for bucket
        timestamp = datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')
        output_filename = f"heygen_4prob_video_{base_name}_{timestamp}.mp4"
        
        # Upload video directly to bucket
        upload_success, bucket_url = upload_video_to_bucket(video_url, output_filename, output_bucket)
        
        if not upload_success:
            raise Exception("Failed to upload video to bucket")
        
        # Calculate metrics
        total_slides = len(slide_content)
        total_chars = sum(len(content) for content in slide_content.values())
        avg_chars = total_chars / total_slides if total_slides > 0 else 0
        
        return {
            "filename": filename,
            "video_id": video_id,
            "video_title": video_title,
            "output_filename": output_filename,
            "bucket_url": bucket_url,
            "video_url": video_url,  # Original HeyGen URL (expires in 7 days)
            "slide_count": total_slides,
            "total_chars": total_chars,
            "avg_chars_per_slide": avg_chars,
            "bucket_name": output_bucket,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"[HEYGEN-4PROB-VIDEO-PROCESS] Error processing {filename}: {str(e)}")
        return {
            "filename": filename,
            "error": str(e),
            "status": "failed"
        }


def process_all_chunked_scripts(template_config: Dict, company_name: str) -> Dict:
    """
    Process all chunked 4-problem scripts in bucket
    
    Args:
        template_config: Template configuration dictionary
        company_name: Company name for filtering files
        
    Returns:
        Summary of all processing results
    """
    logger.info(f"[HEYGEN-4PROB-VIDEO-MAIN] Processing all chunked scripts for company: {company_name}")
    
    results = {
        "company_name": company_name,
        "processed_videos": [],
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "bucket_name": template_config["bucket_name"],
        "output_bucket": template_config["output_bucket"],
        "template_id": template_config["template_id"],
        "timestamp": datetime.now(eastern_tz).strftime("%Y%m%d_%H%M%S")
    }
    
    bucket_name = template_config["bucket_name"]
    
    # List all chunked files in bucket
    all_files = list_files_in_bucket(bucket_name)
    
    # Filter for chunked 4-problem scripts for this company
    chunked_files = [f for f in all_files 
                    if f.startswith(company_name) and 'chunked_4prob_script_' in f]
    
    logger.info(f"[HEYGEN-4PROB-VIDEO-MAIN] Found {len(chunked_files)} chunked script files for {company_name}")
    
    if not chunked_files:
        logger.warning(f"[HEYGEN-4PROB-VIDEO-MAIN] No chunked scripts found for {company_name}. Looking for files starting with '{company_name}' containing 'chunked_4prob_script_'")
        logger.info(f"[HEYGEN-4PROB-VIDEO-MAIN] Available files: {[f for f in all_files if f.startswith(company_name)]}")
    
    # Process each chunked file
    for filename in chunked_files:
        result = process_single_chunked_script(filename, template_config)
        results["processed_videos"].append(result)
        results["total_processed"] += 1
        
        if result["status"] == "success":
            results["successful"] += 1
            logger.info(f"[HEYGEN-4PROB-VIDEO-SUCCESS] Generated video: {result.get('output_filename', 'Unknown')}")
        else:
            results["failed"] += 1
            logger.error(f"[HEYGEN-4PROB-VIDEO-FAILED] Failed to generate video for: {filename}")
    
    logger.info(f"[HEYGEN-4PROB-VIDEO-MAIN] Processing complete: {results['successful']}/{results['total_processed']} videos generated successfully")
    return results

# =====================================================================
# TEST MODE FUNCTIONALITY
# =====================================================================

def test_single_video_generation(test_filename: str = None, template_id: str = None) -> Dict:
    """
    Test function for single video generation - use for initial testing
    
    Args:
        test_filename: Specific chunked script to test with (optional)
        template_id: Override template ID for testing (optional)
    """
    logger.info("[HEYGEN-4PROB-VIDEO-TEST] Running single video generation test...")
    
    try:
        # Use default template config for test
        template_config = TEMPLATE_4PROB_CONFIG.copy()
        
        # Override template ID if provided
        if template_id:
            template_config["template_id"] = template_id
            logger.info(f"[HEYGEN-4PROB-VIDEO-TEST] Using override template ID: {template_id}")
        
        # Template ID is configured and ready
        logger.info(f"[HEYGEN-4PROB-VIDEO-TEST] Using template: {template_config['template_id']}")
        
        logger.info(f"[HEYGEN-4PROB-VIDEO-TEST] Template ID: {template_config['template_id']}")
        
        if test_filename:
            # Test specific file
            logger.info(f"[HEYGEN-4PROB-VIDEO-TEST] Testing with specified file: {test_filename}")
            result = process_single_chunked_script(test_filename, template_config)
        else:
            # Find first available chunked script
            all_files = list_files_in_bucket(template_config["bucket_name"])
            chunked_files = [f for f in all_files if 'chunked_4prob_script_' in f]
            
            if not chunked_files:
                raise Exception(f"No chunked scripts found in bucket {template_config['bucket_name']}")
            
            test_file = chunked_files[0]
            logger.info(f"[HEYGEN-4PROB-VIDEO-TEST] Testing with file: {test_file}")
            result = process_single_chunked_script(test_file, template_config)
        
        return result
        
    except Exception as e:
        logger.error(f"[HEYGEN-4PROB-VIDEO-TEST] Test failed: {str(e)}")
        return {"status": "failed", "error": str(e)}

# =====================================================================
# MAIN FUNCTION WITH TEST MODE
# =====================================================================

def main():
    """Main function to orchestrate the entire HeyGen 4-problem video generation workflow."""
    try:
        print("=" * 80)
        print("HEYGEN 4-PROBLEM VIDEO GENERATOR v4 - SUCCESS PRINCIPLES APPLIED")
        print("=" * 80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Script directory: {SCRIPT_DIR}")
        print(f"Target: Generate videos from chunked 4-problem scripts")
        print(f"Template ID: {TEMPLATE_4PROB_CONFIG['template_id']}")
        print(f"Bucket: {TEMPLATE_4PROB_CONFIG['bucket_name']}")
        print(f"Output Bucket: {TEMPLATE_4PROB_CONFIG['output_bucket']}")
        print()
        print("SUCCESS PRINCIPLES APPLIED:")
        print("✅ Template API endpoint (/v2/template/{id}/generate)")
        print("✅ Correct variable structure (name, type, properties.content)")
        print("✅ Fixed placeholder naming (four_prob_slide_XX_content)")
        print("✅ One slide per scene principle")
        print("✅ Exact payload structure from successful curl")

        # Check for test mode argument
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            print("\n🧪 RUNNING IN TEST MODE - Single video generation")
            print("=" * 50)
            
            test_filename = sys.argv[2] if len(sys.argv) > 2 else None
            template_id = sys.argv[3] if len(sys.argv) > 3 else None
            
            if test_filename:
                print(f"📁 Test file: {test_filename}")
            else:
                print("📁 Test file: Auto-select first available")
            
            if template_id:
                print(f"🎬 Template ID: {template_id}")
            else:
                print(f"🎬 Template ID: {TEMPLATE_4PROB_CONFIG['template_id']}")
            
            result = test_single_video_generation(test_filename, template_id)
            
            print("\n" + "=" * 50)
            print("TEST RESULTS")
            print("=" * 50)
            
            if result["status"] == "success":
                print("✅ Test successful!")
                print(f"   📁 Source: {result['filename']}")
                print(f"   🎬 Video ID: {result['video_id']}")
                print(f"   📄 Video Title: {result['video_title']}")
                print(f"   💾 Output File: {result['output_filename']}")
                print(f"   🔗 Bucket URL: {result['bucket_url']}")
                print(f"   📊 Slides: {result['slide_count']}")
                print(f"   📝 Characters: {result['total_chars']}")
                print(f"   📈 Avg/slide: {result['avg_chars_per_slide']:.1f}")
            else:
                print("❌ Test failed!")
                print(f"   Error: {result.get('error', 'Unknown error')}")
            
            return

        # Continue with normal workflow...
        print("\n🚀 RUNNING IN PRODUCTION MODE - All videos")
        print("=" * 50)

        # Template ID is now configured - ready for production
        logger.info(f"[HEYGEN-4PROB-VIDEO-MAIN] Using configured template: {TEMPLATE_4PROB_CONFIG['template_id']}")

        # Fetch configuration from Supabase
        logger.info("[HEYGEN-4PROB-VIDEO-MAIN] Fetching configuration from Supabase...")
        print("\nFetching configuration from Supabase...")
        variables = fetch_configuration_from_supabase()

        # Validate configuration structure
        if "global" not in variables:
            raise Exception("global configuration not found in Supabase config")

        global_config = variables["global"]
        company_name = global_config.get("COMPANY_NAME")

        if not company_name:
            raise Exception("COMPANY_NAME not found in global configuration")

        # Build template configuration
        template_config = TEMPLATE_4PROB_CONFIG.copy()

        logger.info(f"[HEYGEN-4PROB-VIDEO-MAIN] Using template: {template_config['template_id']}")
        logger.info(f"[HEYGEN-4PROB-VIDEO-MAIN] Processing bucket: {template_config['bucket_name']}")
        logger.info(f"[HEYGEN-4PROB-VIDEO-MAIN] Output bucket: {template_config['output_bucket']}")
        logger.info(f"[HEYGEN-4PROB-VIDEO-MAIN] Company: {company_name}")

        # Process all chunked scripts
        results = process_all_chunked_scripts(template_config, company_name)

        # Display final summary
        print("\n" + "=" * 80)
        print("HEYGEN 4-PROBLEM VIDEO GENERATION SUMMARY")
        print("=" * 80)
        print(f"Company: {results['company_name']}")
        print(f"Total scripts processed: {results['total_processed']}")
        print(f"Videos generated successfully: {results['successful']}")
        print(f"Failed generations: {results['failed']}")
        print(f"Template ID: {results['template_id']}")
        print(f"Source bucket: {results['bucket_name']}")
        print(f"Output bucket: {results['output_bucket']}")

        # Show successful generations
        if results['successful'] > 0:
            print(f"\n✅ Successful video generations:")
            for video in results['processed_videos']:
                if video['status'] == 'success':
                    print(f"  📁 {video['filename']}")
                    print(f"     🎬 Video ID: {video['video_id']}")
                    print(f"     💾 Output: {video['output_filename']}")
                    print(f"     🔗 URL: {video['bucket_url']}")
                    print(f"     📊 {video['slide_count']} slides, {video['total_chars']} chars")

        # Show failed generations
        if results['failed'] > 0:
            print(f"\n❌ Failed video generations:")
            for video in results['processed_videos']:
                if video['status'] == 'failed':
                    print(f"  📁 {video['filename']}")
                    print(f"     ❌ Error: {video.get('error', 'Unknown error')}")

        # Calculate and display execution time
        end_time = datetime.now(eastern_tz)
        execution_time = end_time - start_time
        print(f"\n⏱️ Execution time: {execution_time}")
        print(f"🏁 End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Log successful session completion
        logger.info("=" * 60)
        logger.info("HEYGEN 4-PROBLEM VIDEO GENERATOR v4 - SESSION END (SUCCESS)")
        logger.info("=" * 60)

    except Exception as e:
        logger.critical(f"[HEYGEN-4PROB-VIDEO-CRITICAL] Critical error in video generation workflow: {str(e)}")
        logger.critical(f"[HEYGEN-4PROB-VIDEO-CRITICAL] Traceback: {traceback.format_exc()}")
        print(f"\n❌ Critical error in video generation workflow: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Log session end on failure
        logger.info("=" * 60)
        logger.info("HEYGEN 4-PROBLEM VIDEO GENERATOR v4 - SESSION END (FAILED)")
        logger.info("=" * 60)
        raise


if __name__ == "__main__":
    main()
