#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Script Automation Workflow Script

This script processes Poppy Card content and generates video scripts using:
- Voice guidance (tone and style)
- Method guidance (structure and framework) 
- Prompt instructions (specific processing directions)
- Poppy Card content (unique subject matter)

The workflow processes 10 predefined Poppy Card combinations sequentially,
generating custom video scripts for each combination and saving them to Supabase.

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

# Handle missing environment variables
if not supabase_url or not supabase_service_key:
    print("Error: Missing Supabase credentials")
    exit(1)

supabase = create_client(supabase_url, supabase_service_key)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Record start time for execution tracking (Eastern Time)
eastern_tz = ZoneInfo("America/New_York")
start_time = datetime.now(eastern_tz)

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"video_script_log_{datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}.log")
    ])
logger = logging.getLogger(__name__)

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
        print(f"Downloading {file_name} from {bucket_name} bucket...")
        response = supabase.storage.from_(bucket_name).download(file_name)

        if response:
            content = response.decode('utf-8')
            print(
                f"Successfully downloaded {file_name} ({len(content)} characters)"
            )
            return content
        else:
            raise Exception(f"Failed to download {file_name}")

    except Exception as e:
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
        print(f"Uploading {file_name} to {bucket_name} bucket...")

        # Convert string content to bytes
        file_bytes = file_content.encode('utf-8')

        response = supabase.storage.from_(bucket_name).upload(
            file_name, file_bytes, {"content-type": "text/plain"})

        print(f"Successfully uploaded {file_name} to {bucket_name}")
        return True

    except Exception as e:
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
    Generate a video script using OpenAI API with guidance and Poppy Card content
    
    Args:
        voice_guidance (str): Voice and tone guidance
        method_guidance (str): Script structure and framework guidance
        prompt_instructions (str): Specific processing instructions
        poppy_card_content (str): Poppy Card content to focus on
        openai_model (str): OpenAI model to use
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        str: Generated video script
    """
    try:
        # Create OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)

        # Construct the system prompt using the recommended structure
        system_prompt = f"""You are a professional video script writer. Generate a video script using the following guidance:

VOICE GUIDELINES:
{voice_guidance}

SCRIPT METHOD/FRAMEWORK:
{method_guidance}

SPECIFIC INSTRUCTIONS:
{prompt_instructions}

CONTENT TO FOCUS ON:
{poppy_card_content}

Requirements:
- Write in plain text format
- Use short paragraphs of 1-3 sentences maximum
- Add line breaks between paragraphs
- Create an engaging video script that follows the voice, method, and focuses on the provided content"""

        # Attempt to generate script with retry logic
        for attempt in range(max_retries):
            try:
                print(
                    f"Generating video script using {openai_model} (attempt {attempt + 1}/{max_retries})..."
                )

                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role":
                        "user",
                        "content":
                        "Please generate the video script now."
                    }],
                    max_tokens=2000,
                    temperature=0.7)

                script_content = response.choices[0].message.content
                if script_content:
                    script_content = script_content.strip()

                if script_content:
                    print(
                        f"Successfully generated video script ({len(script_content)} characters)"
                    )
                    return script_content
                else:
                    raise Exception("Empty response from OpenAI")

            except Exception as e:
                error_msg = str(e).lower()

                # Check for model availability issues
                if "model" in error_msg and ("not found" in error_msg
                                             or "unavailable" in error_msg
                                             or "sunset" in error_msg):
                    raise Exception(
                        f"OpenAI model {openai_model} is unavailable or has been sunset. Please update the model configuration."
                    )

                if attempt < max_retries - 1:
                    print(
                        f"OpenAI API error: {e}. Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(
                        f"Failed to generate script after {max_retries} attempts: {e}"
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
    Load the three guidance files from Supabase bucket
    
    Args:
        bucket_name (str): Name of the script-guidance bucket
        
    Returns:
        dict: Dictionary containing voice, method, and prompt guidance
    """
    print("\n" + "=" * 80)
    print("LOADING GUIDANCE FILES")
    print("=" * 80)

    try:
        # Load all three guidance files
        voice_guidance = download_file_from_bucket(bucket_name,
                                                   "voice_guidance.txt")
        method_guidance = download_file_from_bucket(bucket_name,
                                                    "method_guidance.txt")
        prompt_instructions = download_file_from_bucket(
            bucket_name, "prompt_instructions.txt")

        guidance_files = {
            "voice": voice_guidance,
            "method": method_guidance,
            "prompt": prompt_instructions
        }

        print("Successfully loaded all guidance files")
        return guidance_files

    except Exception as e:
        print(f"Error loading guidance files: {str(e)}")
        raise


def process_poppy_cards(variables, guidance_files):
    """
    Process all 10 Poppy Card combinations sequentially
    
    Args:
        variables (dict): Configuration variables from Supabase
        guidance_files (dict): Loaded guidance files
        
    Returns:
        dict: Summary of processed scripts
    """
    print("\n" + "=" * 80)
    print("PROCESSING POPPY CARDS")
    print("=" * 80)

    try:
        # Extract configuration
        global_config = variables["global"]
        video_script_config = variables["scripts"]["video_script"]

        company_name = global_config.get("COMPANY_NAME", "company")
        card_combinations = video_script_config.get("card_combinations", [])
        input_bucket = video_script_config["supabase_buckets"]["input_cards"]
        output_bucket = video_script_config["supabase_buckets"]["output"]
        openai_model = video_script_config.get("openai_model", "gpt-4o")

        # Generate timestamp for output files in Eastern Time (YYYYMMDD_HHMM)
        timestamp = datetime.now(eastern_tz).strftime("%Y%m%d_%H%M")

        processed_scripts = []
        total_cards = len(card_combinations)

        print(
            f"Processing {total_cards} Poppy Card combinations for {company_name}"
        )
        print(f"Using OpenAI model: {openai_model}")
        print(f"Input bucket: {input_bucket}")
        print(f"Output bucket: {output_bucket}")

        # Process each card combination
        for i, combination in enumerate(card_combinations, 1):
            print(f"\nProcessing card {i} of {total_cards}...")
            print(f"Combination: {combination}")

            try:
                # Construct input and output filenames with proper card numbering
                # Card numbers iterate from card01 to card10 to match actual file names
                card_number = f"card{i:02d}"  # Formats as card01, card02, ..., card10
                input_filename = f"{company_name}_{card_number}_{combination}.txt"
                output_filename = f"{company_name}_script_{combination}_{timestamp}.txt"

                print(f"Looking for file: {input_filename}")

                # Download the Poppy Card content
                poppy_card_content = download_file_from_bucket(
                    input_bucket, input_filename)

                # Generate the video script
                script_content = generate_video_script(
                    guidance_files["voice"], guidance_files["method"],
                    guidance_files["prompt"], poppy_card_content, openai_model)

                # Upload the generated script
                upload_file_to_bucket(output_bucket, output_filename,
                                      script_content)

                # Record the processed script
                processed_scripts.append({
                    "combination": combination,
                    "input_file": input_filename,
                    "output_file": output_filename,
                    "script_length": len(script_content),
                    "status": "success"
                })

                print(f"Successfully processed {combination}")

            except Exception as e:
                print(f"Error processing {combination}: {str(e)}")
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
                    "failed"
                })

        # Generate summary
        successful_scripts = [
            s for s in processed_scripts if s["status"] == "success"
        ]
        failed_scripts = [
            s for s in processed_scripts if s["status"] == "failed"
        ]

        summary = {
            "total_processed": total_cards,
            "successful": len(successful_scripts),
            "failed": len(failed_scripts),
            "scripts": processed_scripts,
            "company_name": company_name,
            "timestamp": timestamp,
            "openai_model": openai_model
        }

        print(
            f"\nProcessing complete: {len(successful_scripts)}/{total_cards} scripts generated successfully"
        )

        return summary

    except Exception as e:
        print(f"Error in process_poppy_cards: {str(e)}")
        raise


def main():
    """Main function to orchestrate the entire workflow."""
    try:
        print("=" * 80)
        print("VIDEO SCRIPT AUTOMATION WORKFLOW")
        print("=" * 80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Script directory: {SCRIPT_DIR}")

        # Fetch configuration from Supabase
        print("\nFetching configuration from Supabase...")
        variables = fetch_configuration_from_supabase()

        # Validate that we have the video_script configuration
        if "scripts" not in variables or "video_script" not in variables[
                "scripts"]:
            raise Exception(
                "video_script configuration not found in Supabase config")

        video_script_config = variables["scripts"]["video_script"]

        # Load guidance files
        guidance_bucket = video_script_config["supabase_buckets"]["guidance"]
        guidance_files = load_guidance_files(guidance_bucket)

        # Process Poppy Cards
        summary = process_poppy_cards(variables, guidance_files)

        # Save summary to output bucket
        output_bucket = video_script_config["supabase_buckets"]["output"]
        summary_filename = f"video_script_summary_{summary['timestamp']}.json"
        summary_content = json.dumps(summary, indent=2)
        upload_file_to_bucket(output_bucket, summary_filename, summary_content)

        # Calculate and display execution time
        end_time = datetime.now(eastern_tz)
        execution_time = end_time - start_time

        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETE")
        print("=" * 80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {execution_time}")
        print(
            f"Scripts generated: {summary['successful']}/{summary['total_processed']}"
        )
        print(f"Summary saved as: {summary_filename}")

        if summary['failed'] > 0:
            print(
                f"\nWarning: {summary['failed']} script(s) failed to generate")
            for script in summary['scripts']:
                if script['status'] == 'failed':
                    print(
                        f"  - {script['combination']}: {script.get('error', 'Unknown error')}"
                    )

        print("\nVideo script automation workflow completed successfully!")

    except Exception as e:
        print(f"\nCritical error in main workflow: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
