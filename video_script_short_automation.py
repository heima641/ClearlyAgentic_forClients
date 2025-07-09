#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Script SHORT Automation Workflow Script

This script processes Poppy Card content and generates SHORT video scripts using:
- Voice guidance (tone and style)
- Method guidance (structure and framework) 
- Prompt instructions (specific processing directions for 5-minute format)
- Poppy Card content (unique subject matter for 2-problem cards)

The workflow processes 5 predefined Poppy Card combinations sequentially (cards 11-15),
generating custom SHORT video scripts for each combination and saving them to Supabase.

Card Range: 11-15 (specifically for 2-problem format cards)
Target Duration: 5 minutes
Output Prefix: SHORT

All output files are saved to Supabase Storage buckets.
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
from datetime import datetime
from zoneinfo import ZoneInfo

# Import OpenAI for script generation
import openai

# SUP BUCKET OUTPUT SUP BUCKET OUTPUT
from dotenv import load_dotenv
from supabase import create_client  # # SUPABASE BUCKET: import Supabase client
from supabase import Client

# =====================================================================
# CONSTANTS AND CONFIGURATION
# =====================================================================

# Script directory for use in file path construction
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to environment variables file
ENV_FILE_PATH = "STATIC_VARS_MAR2025.env"

# SUP BUCKET OUTPUT SUP BUCKET OUTPUT
# Load environment variables (for any additional env vars needed)
load_dotenv(ENV_FILE_PATH)
supabase_url = os.getenv("VITE_SUPABASE_URL")
supabase_service_key = os.getenv("VITE_SUPABASE_SERVICE_ROLE_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Handle missing environment variables - NO FALLBACKS
if not supabase_url:
    print("CRITICAL ERROR: VITE_SUPABASE_URL environment variable is required but not found")
    print("Please ensure VITE_SUPABASE_URL is set in your STATIC_VARS_MAR2025.env file")
    exit(1)

if not supabase_service_key:
    print("CRITICAL ERROR: VITE_SUPABASE_SERVICE_ROLE_KEY environment variable is required but not found")
    print("Please ensure VITE_SUPABASE_SERVICE_ROLE_KEY is set in your STATIC_VARS_MAR2025.env file")
    exit(1)

if not openai_api_key:
    print("CRITICAL ERROR: OPENAI_API_KEY environment variable is required but not found")
    print("Please ensure OPENAI_API_KEY is set in your STATIC_VARS_MAR2025.env file")
    exit(1)

supabase = create_client(supabase_url, supabase_service_key)

# Record start time for execution tracking (Eastern Time)
eastern_tz = ZoneInfo("America/New_York")
start_time = datetime.now(eastern_tz)

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # seconds

# Configure logging with SHORT identifier
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SHORT-SCRIPT] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"video_script_SHORT_log_{datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}.log")
    ])
logger = logging.getLogger(__name__)

# Log initialization for SHORT workflow
logger.info("=" * 60)
logger.info("SHORT VIDEO SCRIPT AUTOMATION - SESSION START")
logger.info(f"Target: 5-minute scripts from cards 11-15")
logger.info(f"Session ID: {datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}")
logger.info("=" * 60)

# Suppress excessive HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Setup directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_PATH = os.path.join(script_dir, "STATIC_VARS_MAR2025.env")

# =====================================================================
# COMMON FUNCTIONS
# =====================================================================


def fetch_configuration_from_supabase(config_name=None, config_id=None):
    """
    Fetch configuration variables from Supabase workflow_configs table

    Args:
        config_name (str, optional): Name of the configuration to fetch
        config_id (int, optional): ID of the configuration to fetch

    Returns:
        dict: Configuration variables
    """
    try:
        # Use the global Supabase client
        global supabase

        # Query based on either name or ID
        if config_id:
            print(f"Fetching configuration with ID: {config_id}")
            response = supabase.table("workflow_configs").select("*").eq(
                "id", config_id).execute()
        elif config_name:
            print(f"Fetching configuration with name: {config_name}")
            response = supabase.table("workflow_configs").select("*").eq(
                "config_name", config_name).execute()
        else:
            # If no specific config requested, get the most recent one
            print("Fetching most recent configuration")
            response = supabase.table("workflow_configs").select("*").order(
                "created_at", desc=True).limit(1).execute()

        # Check if we got any data
        if not response.data or len(response.data) == 0:
            print("No configuration found in Supabase")
            raise Exception("No configuration found in Supabase")

        # Return the variables from the first matching record
        config_data = response.data[0]
        print(
            f"Successfully fetched configuration: {config_data.get('config_name', 'unnamed')}"
        )
        return config_data.get("variables", {})

    except Exception as e:
        print(f"Error fetching configuration from Supabase: {str(e)}")
        raise


def download_file_from_bucket(bucket_name, file_name):
    """
    Download a file from Supabase storage bucket
    
    Args:
        bucket_name (str): Name of the Supabase bucket
        file_name (str): Name of the file to download
        
    Returns:
        str: File content as string
    """
    try:
        logger.info(f"[SHORT-WORKFLOW] Downloading {file_name} from {bucket_name} bucket...")
        print(f"Downloading {file_name} from {bucket_name} bucket...")
        response = supabase.storage.from_(bucket_name).download(file_name)

        if response:
            content = response.decode('utf-8')
            logger.info(f"[SHORT-WORKFLOW] Successfully downloaded {file_name} ({len(content)} characters)")
            print(
                f"Successfully downloaded {file_name} ({len(content)} characters)"
            )
            return content
        else:
            raise Exception(f"Failed to download {file_name}")

    except Exception as e:
        logger.error(f"[SHORT-WORKFLOW] Error downloading {file_name} from {bucket_name}: {str(e)}")
        print(f"Error downloading {file_name} from {bucket_name}: {str(e)}")
        raise


def upload_file_to_bucket(bucket_name, file_name, file_content):
    """
    Upload a file to Supabase storage bucket
    
    Args:
        bucket_name (str): Name of the Supabase bucket
        file_name (str): Name of the file to upload
        file_content (str): Content to upload
        
    Returns:
        bool: True if successful
    """
    try:
        logger.info(f"[SHORT-WORKFLOW] Uploading {file_name} to {bucket_name} bucket...")
        print(f"Uploading {file_name} to {bucket_name} bucket...")

        # Convert string content to bytes
        file_bytes = file_content.encode('utf-8')

        response = supabase.storage.from_(bucket_name).upload(
            file_name, file_bytes, {"content-type": "text/plain"})

        logger.info(f"[SHORT-WORKFLOW] Successfully uploaded {file_name} to {bucket_name}")
        print(f"Successfully uploaded {file_name} to {bucket_name}")
        return True

    except Exception as e:
        logger.error(f"[SHORT-WORKFLOW] Error uploading {file_name} to {bucket_name}: {str(e)}")
        print(f"Error uploading {file_name} to {bucket_name}: {str(e)}")
        raise


def generate_video_script(voice_guidance,
                          method_guidance,
                          prompt_instructions,
                          poppy_card_content,
                          openai_model="gpt-4o",
                          max_retries=3,
                          retry_delay=2):
    """
    Generate a SHORT video script using OpenAI API with guidance and Poppy Card content
    
    Args:
        voice_guidance (str): Voice and tone guidance
        method_guidance (str): Script structure and framework guidance
        prompt_instructions (str): Specific processing instructions for 5-minute format
        poppy_card_content (str): Poppy Card content to focus on (2-problem format)
        openai_model (str): OpenAI model to use
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        str: Generated SHORT video script
    """
    try:
        # Create OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)

        # Construct the optimized system prompt for 5-minute, 2-problem format
        system_prompt = f"""You are a professional video script writer specializing in concise, high-impact content. Generate a SHORT video script with these precise specifications:

TARGET: 5-minute duration (approximately 750-900 words)
FORMAT: 2-problem structure for maximum engagement

VOICE & TONE GUIDELINES:
{voice_guidance}

STRUCTURAL FRAMEWORK (Adapted for Short Format):
{method_guidance}

SPECIFIC SHORT-FORMAT INSTRUCTIONS:
{prompt_instructions}

CONTENT SOURCE (2-Problem Card):
{poppy_card_content}

CRITICAL SHORT-FORMAT REQUIREMENTS:
1. DURATION CONTROL: Aim for 5 minutes (750-900 words maximum)
2. 2-PROBLEM STRUCTURE: 
   - Problem 1: 2-2.5 minutes of content
   - Transition: 15-30 seconds  
   - Problem 2: 2-2.5 minutes of content
3. PACING: Fast, dynamic transitions between concepts
4. DENSITY: Pack maximum value into minimum time
5. KALLAWAY FRAMEWORK ADAPTATION:
   - Hook: 15-20 seconds (immediate engagement)
   - Authority: Integrated throughout, not separate section
   - Logic: Streamlined, essential points only
   - Leverage: Quick, actionable insights
   - Appeal: Concise call-to-action
   - Why: Woven into problems, not standalone
6. FORMATTING:
   - Short paragraphs (1-3 sentences max)
   - Clear section breaks
   - Rapid-fire delivery style
   - No filler content

OPTIMIZATION TACTICS:
- Lead with strongest problem first
- Use power words and active voice
- Eliminate redundancy and weak transitions
- Focus on outcomes and results
- Create urgency without rushing
- End each problem with clear takeaway

Generate a script that delivers maximum impact in minimum time while maintaining professional quality and clear structure."""

        # Attempt to generate script with enhanced retry logic for SHORT scripts
        for attempt in range(max_retries):
            try:
                logger.info(f"[SHORT-GENERATION] Attempt {attempt + 1}/{max_retries} for 5-minute script")
                print(
                    f"Generating 5-minute SHORT video script using {openai_model} (attempt {attempt + 1}/{max_retries})..."
                )

                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role": "user",
                        "content": "Generate a 5-minute SHORT video script now. Focus on the 2-problem structure from the provided card content. Ensure tight pacing, maximum value density, and clear transitions between problems. Target 750-900 words total."
                    }],
                    max_tokens=2000,
                    temperature=0.7)

                script_content = response.choices[0].message.content
                if script_content:
                    script_content = script_content.strip()

                if script_content:
                    # Calculate approximate word count for 5-minute target validation
                    word_count = len(script_content.split())
                    logger.info(f"[SHORT-GENERATION] SUCCESS - Generated {word_count} words, {len(script_content)} characters")
                    print(
                        f"Successfully generated 5-minute SHORT video script ({len(script_content)} characters, ~{word_count} words)"
                    )
                    
                    # Log if word count is outside optimal range for 5-minute content
                    if word_count < 700:
                        logger.warning(f"[SHORT-VALIDATION] Word count ({word_count}) below 5-minute target (700-950)")
                        print(f"  → Note: Word count ({word_count}) may be low for 5-minute target")
                    elif word_count > 950:
                        logger.warning(f"[SHORT-VALIDATION] Word count ({word_count}) above 5-minute target (700-950)")
                        print(f"  → Note: Word count ({word_count}) may be high for 5-minute target")
                    else:
                        logger.info(f"[SHORT-VALIDATION] Word count ({word_count}) within 5-minute target range")
                        print(f"  → Word count ({word_count}) is within 5-minute target range")
                    
                    return script_content
                else:
                    raise Exception("Empty response from OpenAI for SHORT script generation")

            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"[SHORT-ERROR] Attempt {attempt + 1} failed: {str(e)}")

                # Enhanced error handling for SHORT script context
                if "model" in error_msg and ("not found" in error_msg
                                             or "unavailable" in error_msg
                                             or "sunset" in error_msg):
                    logger.critical(f"[SHORT-CRITICAL] OpenAI model {openai_model} unavailable for SHORT script generation")
                    raise Exception(
                        f"OpenAI model {openai_model} is unavailable or has been sunset. Please update the SHORT script model configuration."
                    )

                # Rate limiting specific handling for SHORT scripts
                if "rate" in error_msg and "limit" in error_msg:
                    logger.warning(f"[SHORT-RATE-LIMIT] Hit rate limit during SHORT script generation")
                    if attempt < max_retries - 1:
                        extended_delay = retry_delay * 3  # Longer delay for rate limits
                        logger.info(f"[SHORT-RETRY] Rate limit - waiting {extended_delay}s before retry")
                        time.sleep(extended_delay)
                        continue

                # Missing cards 11-15 specific error handling
                if "not found" in error_msg or "404" in error_msg:
                    logger.error(f"[SHORT-MISSING-CARD] Card file not found - likely missing card in 11-15 range")

                if attempt < max_retries - 1:
                    logger.info(f"[SHORT-RETRY] Retrying SHORT script generation in {retry_delay}s...")
                    print(
                        f"OpenAI API error: {e}. Retrying SHORT script in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"[SHORT-FAILURE] Failed to generate SHORT script after {max_retries} attempts")
                    print(
                        f"Failed to generate SHORT script after {max_retries} attempts: {e}"
                    )
                    raise

    except Exception as e:
        print(f"Error in generate_video_script: {str(e)}")
        raise


# =====================================================================
# MAIN WORKFLOW FUNCTIONS
# =====================================================================


def load_guidance_files(bucket_name):
    """
    Load the three guidance files from Supabase bucket (SHORT version)
    
    Args:
        bucket_name (str): Name of the script-guidance bucket
        
    Returns:
        dict: Dictionary containing voice, method, and SHORT prompt guidance
    """
    print("\n" + "=" * 80)
    print("LOADING GUIDANCE FILES FOR SHORT SCRIPTS")
    print("=" * 80)

    try:
        # Load all three guidance files - using SHORT version of prompt instructions
        # NO FALLBACKS - all files must exist or the process fails
        voice_guidance = download_file_from_bucket(bucket_name,
                                                   "voice_guidance.txt")
        method_guidance = download_file_from_bucket(bucket_name,
                                                    "method_guidance.txt")
        prompt_instructions = download_file_from_bucket(
            bucket_name, "prompt_shorter_instructions.txt")  # Using shorter version

        # Validate that all guidance files have content
        if not voice_guidance or not voice_guidance.strip():
            raise Exception("voice_guidance.txt is empty or contains only whitespace")
        
        if not method_guidance or not method_guidance.strip():
            raise Exception("method_guidance.txt is empty or contains only whitespace")
        
        if not prompt_instructions or not prompt_instructions.strip():
            raise Exception("prompt_shorter_instructions.txt is empty or contains only whitespace")

        guidance_files = {
            "voice": voice_guidance,
            "method": method_guidance,
            "prompt": prompt_instructions
        }

        print("Successfully loaded all guidance files for SHORT scripts")
        return guidance_files

    except Exception as e:
        print(f"Error loading guidance files: {str(e)}")
        raise Exception(f"Failed to load required guidance files from bucket '{bucket_name}': {str(e)}")


def process_poppy_cards(variables, guidance_files, company_name, card_combinations, input_bucket, output_bucket, openai_model):
    """
    Process 5 Poppy Card combinations sequentially (cards 11-15)
    
    Args:
        variables (dict): Configuration variables from Supabase
        guidance_files (dict): Loaded guidance files
        company_name (str): Company name (validated, no fallbacks)
        card_combinations (list): List of card combinations (validated, no fallbacks)
        input_bucket (str): Input bucket name (validated, no fallbacks)
        output_bucket (str): Output bucket name (validated, no fallbacks)
        openai_model (str): OpenAI model name (validated, no fallbacks)
        
    Returns:
        dict: Summary of processed SHORT scripts
    """
    print("\n" + "=" * 80)
    print("PROCESSING POPPY CARDS (SHORT SCRIPTS - CARDS 11-15)")
    print("=" * 80)

    try:
        # Generate timestamp for output files in Eastern Time (YYYYMMDD_HHMM)
        timestamp = datetime.now(eastern_tz).strftime("%Y%m%d_%H%M")

        processed_scripts = []
        total_cards = len(card_combinations)

        print(
            f"Processing {total_cards} SHORT Poppy Card combinations for {company_name}"
        )
        print(f"Card range: 11-15 (2-problem format)")
        print(f"Using OpenAI model: {openai_model}")
        print(f"Input bucket: {input_bucket}")
        print(f"Output bucket: {output_bucket}")

        # Process each card combination for cards 11-15
        for i, combination in enumerate(card_combinations, 1):
            card_progress = f"[{i}/{total_cards}]"
            logger.info(f"[SHORT-PROGRESS] {card_progress} Starting processing for card {i+10} - {combination}")
            print(f"\nProcessing SHORT card {i} of {total_cards}...")
            print(f"Combination: {combination}")

            try:
                # Construct input and output filenames for cards 11-15
                # Card numbers iterate from card11 to card15 to match actual file names
                card_number = f"card{i+10:02d}"  # Formats as card11, card12, ..., card15
                input_filename = f"{company_name}_{card_number}_{combination}.txt"
                output_filename = f"{company_name}_SHORT_script_{combination}_{timestamp}.txt"  # Added SHORT prefix

                logger.info(f"[SHORT-PROGRESS] {card_progress} Input: {input_filename}")
                logger.info(f"[SHORT-PROGRESS] {card_progress} Output: {output_filename}")
                print(f"Looking for file: {input_filename}")

                # Download the Poppy Card content
                poppy_card_content = download_file_from_bucket(
                    input_bucket, input_filename)

                # Generate the SHORT video script
                logger.info(f"[SHORT-PROGRESS] {card_progress} Generating 5-minute script for {combination}")
                script_content = generate_video_script(
                    guidance_files["voice"], guidance_files["method"],
                    guidance_files["prompt"], poppy_card_content, openai_model)

                # Calculate word count for 5-minute validation
                word_count = len(script_content.split())

                # Upload the generated script
                upload_file_to_bucket(output_bucket, output_filename,
                                      script_content)

                # Record the processed script
                processed_scripts.append({
                    "combination": combination,
                    "input_file": input_filename,
                    "output_file": output_filename,
                    "script_length": len(script_content),
                    "word_count": word_count,  # Added word count tracking
                    "target_compliance": "optimal" if 700 <= word_count <= 950 else ("low" if word_count < 700 else "high"),
                    "status": "success",
                    "script_type": "SHORT"  # Added script type identifier
                })

                logger.info(f"[SHORT-SUCCESS] {card_progress} Generated {word_count} words for {combination}")
                print(f"Successfully processed SHORT {combination}")

            except Exception as e:
                logger.error(f"[SHORT-FAILURE] {card_progress} Failed processing {combination}: {str(e)}")
                print(f"Error processing SHORT {combination}: {str(e)}")
                processed_scripts.append({
                    "combination":
                    combination,
                    "input_file":
                    input_filename
                    if 'input_filename' in locals() else "unknown",
                    "output_file":
                    output_filename
                    if 'output_filename' in locals() else "unknown",
                    "error":
                    str(e),
                    "status":
                    "failed",
                    "script_type": "SHORT"
                })

        # Generate summary with word count analysis
        successful_scripts = [
            s for s in processed_scripts if s["status"] == "success"
        ]
        failed_scripts = [
            s for s in processed_scripts if s["status"] == "failed"
        ]

        # Calculate word count statistics for 5-minute compliance
        if successful_scripts:
            word_counts = [s["word_count"] for s in successful_scripts]
            avg_word_count = sum(word_counts) / len(word_counts)
            optimal_count = len([s for s in successful_scripts if s["target_compliance"] == "optimal"])
        else:
            avg_word_count = 0
            optimal_count = 0

        summary = {
            "total_processed": total_cards,
            "successful": len(successful_scripts),
            "failed": len(failed_scripts),
            "scripts": processed_scripts,
            "company_name": company_name,
            "timestamp": timestamp,
            "openai_model": openai_model,
            "script_type": "SHORT",  # Added script type identifier
            "card_range": "11-15",   # Added card range identifier
            "target_duration": "5 minutes",
            "word_count_analysis": {
                "average_word_count": round(avg_word_count, 1),
                "optimal_compliance": f"{optimal_count}/{len(successful_scripts)}" if successful_scripts else "0/0",
                "target_range": "700-950 words"
            }
        }

        print(
            f"\nSHORT script processing complete: {len(successful_scripts)}/{total_cards} scripts generated successfully"
        )
        
        # Report word count compliance for 5-minute target
        if successful_scripts:
            avg_words = summary["word_count_analysis"]["average_word_count"]
            optimal_ratio = summary["word_count_analysis"]["optimal_compliance"]
            print(f"Word count analysis: Average {avg_words} words per script")
            print(f"5-minute target compliance: {optimal_ratio} scripts in optimal range (700-950 words)")

        return summary

    except Exception as e:
        print(f"Error in process_poppy_cards: {str(e)}")
        raise


def main():
    """Main function to orchestrate the entire SHORT video script workflow."""
    try:
        logger.info("=" * 80)
        logger.info("SHORT VIDEO SCRIPT AUTOMATION WORKFLOW - MAIN START")
        logger.info("=" * 80)
        print("=" * 80)
        print("SHORT VIDEO SCRIPT AUTOMATION WORKFLOW")
        print("=" * 80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Script directory: {SCRIPT_DIR}")
        print(f"Target: 5-minute scripts from cards 11-15")

        # Fetch configuration from Supabase with error handling
        logger.info("[SHORT-CONFIG] Fetching configuration from Supabase...")
        print("\nFetching SHORT configuration from Supabase...")
        
        try:
            variables = fetch_configuration_from_supabase()
        except Exception as e:
            logger.critical(f"[SHORT-CRITICAL] Failed to fetch configuration: {str(e)}")
            raise Exception(f"Failed to fetch SHORT script configuration from Supabase: {str(e)}")

        # Validate that we have the video_script_short configuration
        if "scripts" not in variables or "video_script_short" not in variables["scripts"]:
            logger.critical("[SHORT-CRITICAL] Missing video_script_short configuration in Supabase")
            raise Exception(
                "video_script_short configuration not found in Supabase config. Please ensure the configuration includes a 'video_script_short' section.")

        # Validate global configuration
        if "global" not in variables:
            logger.critical("[SHORT-CRITICAL] Missing global configuration in Supabase")
            raise Exception("global configuration not found in Supabase config. Please ensure the configuration includes a 'global' section.")
        
        global_config = variables["global"]
        if "COMPANY_NAME" not in global_config:
            logger.critical("[SHORT-CRITICAL] Missing COMPANY_NAME in global configuration")
            raise Exception("COMPANY_NAME not found in global configuration. Please ensure the configuration includes a valid COMPANY_NAME.")
        
        company_name = global_config["COMPANY_NAME"]
        if not company_name or not company_name.strip():
            logger.critical("[SHORT-CRITICAL] Empty COMPANY_NAME in configuration")
            raise Exception("COMPANY_NAME is empty or contains only whitespace. Please ensure the configuration includes a valid company name.")

        video_script_config = variables["scripts"]["video_script_short"]
        logger.info("[SHORT-CONFIG] Successfully loaded video_script_short configuration")

        # Validate required configuration structure
        if "supabase_buckets" not in video_script_config:
            logger.critical("[SHORT-CRITICAL] Missing supabase_buckets in video_script_short configuration")
            raise Exception("supabase_buckets not found in video_script_short configuration. Please ensure the configuration includes bucket definitions.")
        
        bucket_config = video_script_config["supabase_buckets"]
        required_buckets = ["input_cards", "guidance", "output"]
        for bucket_name in required_buckets:
            if bucket_name not in bucket_config:
                logger.critical(f"[SHORT-CRITICAL] Missing {bucket_name} bucket in configuration")
                raise Exception(f"{bucket_name} bucket not found in supabase_buckets configuration. Please ensure all required buckets are defined.")

        # Validate card combinations configuration
        if "card_combinations" not in video_script_config:
            logger.critical("[SHORT-CRITICAL] Missing card_combinations in video_script_short configuration")
            raise Exception("card_combinations not found in video_script_short configuration. Please ensure the configuration includes card_combinations array.")
        
        card_combinations = video_script_config["card_combinations"]
        if not card_combinations:
            logger.critical("[SHORT-CRITICAL] Empty card_combinations array in configuration")
            raise Exception("card_combinations array is empty. Please ensure the configuration includes exactly 5 card combinations for cards 11-15.")
        
        if len(card_combinations) != 5:
            logger.critical(f"[SHORT-CRITICAL] Expected 5 card combinations, found {len(card_combinations)}")
            raise Exception(f"Expected exactly 5 card combinations for cards 11-15, found {len(card_combinations)}. Please ensure the configuration includes exactly 5 combinations.")

        # Validate OpenAI model configuration  
        if "openai_model" not in video_script_config:
            logger.critical("[SHORT-CRITICAL] Missing openai_model in video_script_short configuration")
            raise Exception("openai_model not found in video_script_short configuration. Please ensure the configuration includes a valid openai_model.")
        
        openai_model = video_script_config["openai_model"]
        if not openai_model or not openai_model.strip():
            logger.critical("[SHORT-CRITICAL] Empty openai_model in configuration")
            raise Exception("openai_model is empty or contains only whitespace. Please ensure the configuration includes a valid OpenAI model name.")

        # Load guidance files (using SHORT version) with no fallback handling
        if "guidance" not in bucket_config:
            logger.critical("[SHORT-CRITICAL] Missing guidance bucket in configuration")
            raise Exception("guidance bucket not found in supabase_buckets configuration.")
        
        guidance_bucket = bucket_config["guidance"]
        try:
            guidance_files = load_guidance_files(guidance_bucket)
            logger.info("[SHORT-CONFIG] Successfully loaded all guidance files")
        except Exception as e:
            logger.critical(f"[SHORT-CRITICAL] Failed to load guidance files: {str(e)}")
            raise Exception(f"Failed to load SHORT script guidance files from bucket '{guidance_bucket}': {str(e)}")

        # Process Poppy Cards (cards 11-15) with comprehensive error handling
        try:
            logger.info("[SHORT-PROCESSING] Starting card processing for cards 11-15")
            summary = process_poppy_cards(variables, guidance_files, company_name, card_combinations, bucket_config["input_cards"], bucket_config["output"], openai_model)
            logger.info(f"[SHORT-PROCESSING] Completed processing: {summary['successful']}/{summary['total_processed']} successful")
        except Exception as e:
            logger.critical(f"[SHORT-CRITICAL] Failed during card processing: {str(e)}")
            raise Exception(f"Failed to process SHORT script cards 11-15: {str(e)}")

        # Save summary to output bucket - MUST SUCCEED  
        summary_filename = f"video_script_SHORT_summary_{summary['timestamp']}.json"
        summary_content = json.dumps(summary, indent=2)
        
        try:
            upload_file_to_bucket(bucket_config["output"], summary_filename, summary_content)
            logger.info(f"[SHORT-SUMMARY] Saved summary as {summary_filename}")
        except Exception as e:
            logger.error(f"[SHORT-ERROR] Failed to save summary: {str(e)}")
            # Don't raise here - workflow succeeded even if summary save failed
            print(f"Warning: Could not save summary file to bucket '{bucket_config['output']}': {str(e)}")
            print("Video script generation completed successfully despite summary save failure.")

        # Calculate and display execution time
        end_time = datetime.now(eastern_tz)
        execution_time = end_time - start_time

        logger.info("=" * 80)
        logger.info("SHORT VIDEO SCRIPT WORKFLOW COMPLETE")
        logger.info(f"Execution time: {execution_time}")
        logger.info(f"Success rate: {summary['successful']}/{summary['total_processed']}")
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("SHORT VIDEO SCRIPT WORKFLOW COMPLETE")
        print("=" * 80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {execution_time}")
        print(
            f"SHORT scripts generated: {summary['successful']}/{summary['total_processed']}"
        )
        print(f"Card range processed: {summary['card_range']}")
        print(f"Summary saved as: {summary_filename}")

        if summary['failed'] > 0:
            logger.warning(f"[SHORT-WARNING] {summary['failed']} scripts failed to generate")
            print(
                f"\nWarning: {summary['failed']} SHORT script(s) failed to generate")
            for script in summary['scripts']:
                if script['status'] == 'failed':
                    logger.error(f"[SHORT-FAILED-SCRIPT] {script['combination']}: {script.get('error', 'Unknown error')}")
                    print(
                        f"  - {script['combination']}: {script.get('error', 'Unknown error')}"
                    )

        print("\nSHORT video script automation workflow completed successfully!")
        
        # Log successful session completion
        logger.info("=" * 60)
        logger.info("SHORT VIDEO SCRIPT AUTOMATION - SESSION END (SUCCESS)")
        logger.info("=" * 60)

    except Exception as e:
        logger.critical(f"[SHORT-CRITICAL] Critical error in SHORT workflow: {str(e)}")
        logger.critical(f"[SHORT-CRITICAL] Traceback: {traceback.format_exc()}")
        print(f"\nCritical error in SHORT workflow: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Ensure we log session end even on failure
        logger.info("=" * 60)
        logger.info("SHORT VIDEO SCRIPT AUTOMATION - SESSION END (FAILED)")
        logger.info("=" * 60)
        raise


if __name__ == "__main__":
    main()
