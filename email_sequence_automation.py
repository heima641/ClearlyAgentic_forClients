#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Email Sequence Automation Workflow Script

This script processes PSQ Card content and generates 9-email sequences using:
- B2B Software Email Sequence Style Guide (guidance)
- Example Company Software 9-Email Sequence (template)
- PSQ Card content (unique subject matter for problems/solutions/quotes)

The workflow processes 15 PSQ Card files sequentially,
generating custom 9-email sequences for each and saving them to Supabase.

Card Distribution:
- Cards 1-10 (4-problem format) → 'four-prob-emails' bucket
- Cards 11-15 (2-problem format) → 'two-prob-emails' bucket

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

# Import OpenAI for email sequence generation
import openai

# Supabase integration
from dotenv import load_dotenv
from supabase import create_client
from supabase import Client

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
openai_api_key = os.getenv("OPENAI_API_KEY")

# Handle missing environment variables
if not supabase_url:
    print("CRITICAL ERROR: VITE_SUPABASE_URL environment variable is required but not found")
    exit(1)

if not supabase_service_key:
    print("CRITICAL ERROR: VITE_SUPABASE_SERVICE_ROLE_KEY environment variable is required but not found")
    exit(1)

if not openai_api_key:
    print("CRITICAL ERROR: OPENAI_API_KEY environment variable is required but not found")
    exit(1)

supabase = create_client(supabase_url, supabase_service_key)

# Record start time for execution tracking (Eastern Time)
eastern_tz = ZoneInfo("America/New_York")
start_time = datetime.now(eastern_tz)

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # seconds

# Configure logging with EMAIL identifier
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [EMAIL-SEQ] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"email_sequence_log_{datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}.log")
    ])
logger = logging.getLogger(__name__)

# Log initialization for EMAIL workflow
logger.info("=" * 60)
logger.info("EMAIL SEQUENCE AUTOMATION - SESSION START")
logger.info(f"Target: 9-email sequences from 15 PSQ cards")
logger.info(f"Session ID: {datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}")
logger.info("=" * 60)

# Suppress excessive HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

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
        global supabase

        if config_id:
            print(f"Fetching configuration with ID: {config_id}")
            response = supabase.table("workflow_configs").select("*").eq(
                "id", config_id).execute()
        elif config_name:
            print(f"Fetching configuration with name: {config_name}")
            response = supabase.table("workflow_configs").select("*").eq(
                "config_name", config_name).execute()
        else:
            print("Fetching most recent configuration")
            response = supabase.table("workflow_configs").select("*").order(
                "created_at", desc=True).limit(1).execute()

        if not response.data or len(response.data) == 0:
            print("No configuration found in Supabase")
            raise Exception("No configuration found in Supabase")

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
        logger.info(f"[EMAIL-WORKFLOW] Downloading {file_name} from {bucket_name} bucket...")
        print(f"Downloading {file_name} from {bucket_name} bucket...")
        response = supabase.storage.from_(bucket_name).download(file_name)

        if response:
            content = response.decode('utf-8')
            logger.info(f"[EMAIL-WORKFLOW] Successfully downloaded {file_name} ({len(content)} characters)")
            print(f"Successfully downloaded {file_name} ({len(content)} characters)")
            return content
        else:
            raise Exception(f"Failed to download {file_name}")

    except Exception as e:
        logger.error(f"[EMAIL-WORKFLOW] Error downloading {file_name} from {bucket_name}: {str(e)}")
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
        logger.info(f"[EMAIL-WORKFLOW] Uploading {file_name} to {bucket_name} bucket...")
        print(f"Uploading {file_name} to {bucket_name} bucket...")

        # Convert string content to bytes
        file_bytes = file_content.encode('utf-8')

        response = supabase.storage.from_(bucket_name).upload(
            file_name, file_bytes, {"content-type": "text/plain"})

        logger.info(f"[EMAIL-WORKFLOW] Successfully uploaded {file_name} to {bucket_name}")
        print(f"Successfully uploaded {file_name} to {bucket_name}")
        return True

    except Exception as e:
        logger.error(f"[EMAIL-WORKFLOW] Error uploading {file_name} to {bucket_name}: {str(e)}")
        print(f"Error uploading {file_name} to {bucket_name}: {str(e)}")
        raise


def generate_email_sequence(style_guide, example_sequence, psq_card_content, openai_model="gpt-4o", max_retries=3, retry_delay=2):
    """
    Generate a 9-email sequence using OpenAI API with style guide, example, and PSQ card content
    
    Args:
        style_guide (str): B2B Software Email Sequence Style Guide content
        example_sequence (str): Example Company Software 9-Email Sequence content
        psq_card_content (str): PSQ Card content (Problems, Solutions, Quotes)
        openai_model (str): OpenAI model to use
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        str: Generated 9-email sequence
    """
    try:
        # Create OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)

        # Construct the system prompt for email sequence generation
        system_prompt = f"""You are a professional B2B email sequence writer specializing in software marketing. Generate a complete 9-email sequence following these specifications:

STYLE GUIDE REQUIREMENTS:
{style_guide}

EXAMPLE SEQUENCE FOR REFERENCE (follow this structure and style):
{example_sequence}

PSQ CARD CONTENT TO INTEGRATE:
{psq_card_content}

CRITICAL REQUIREMENTS:
1. SEQUENCE LENGTH: Exactly 9 emails
2. EMAIL STRUCTURE: Follow the 3-phase progression (Mindset Priming, Social Proof & Authority, Conversion & Urgency)
3. PSQ INTEGRATION: 
   - Weave the Problems from the PSQ card throughout emails 1-3
   - Use Solutions from PSQ card in emails 4-6 as case studies and social proof
   - Integrate Quotes from PSQ card naturally throughout the sequence (2 quotes per email)
4. FORMATTING:
   - Each email should be clearly numbered (Email 1, Email 2, etc.)
   - Include subject lines for each email
   - Include send timing (Day 1, Day 3, etc.)
   - Maintain conversational, psychology-first tone
   - Use customer transformation focus, not feature focus

EMAIL DISTRIBUTION REQUIREMENTS:
- Email 1-3: Challenge conventional wisdom using PSQ Problems, address limiting beliefs
- Email 4-6: Build authority using PSQ Solutions as case studies, include PSQ Quotes as social proof
- Email 7-9: Drive conversion with urgency, address objections, include guarantee language

QUOTE INTEGRATION:
- Use exactly 2 quotes from the PSQ card per email
- Integrate quotes naturally into the narrative flow
- Format quotes as: "[Customer/Company] told me: '[Quote text]'"
- Distribute all PSQ quotes across the 9 emails (18 total quote placements)

OUTPUT FORMAT:
Generate a complete, ready-to-use 9-email sequence that follows the exact structure of the example sequence while incorporating the PSQ card content seamlessly."""

        # Attempt to generate email sequence with enhanced retry logic
        for attempt in range(max_retries):
            try:
                logger.info(f"[EMAIL-GENERATION] Attempt {attempt + 1}/{max_retries} for 9-email sequence")
                print(f"Generating 9-email sequence using {openai_model} (attempt {attempt + 1}/{max_retries})...")

                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role": "user",
                        "content": "Generate a complete 9-email sequence now. Follow the style guide requirements, use the example sequence as a structural template, and seamlessly integrate the PSQ card content throughout. Ensure each email has proper subject lines, timing, and includes exactly 2 quotes from the PSQ card content."
                    }],
                    max_tokens=4000,  # Increased for full 9-email sequence
                    temperature=0.7)

                sequence_content = response.choices[0].message.content
                if sequence_content:
                    sequence_content = sequence_content.strip()

                if sequence_content:
                    # Validate that the sequence contains 9 emails
                    email_count = len(re.findall(r'Email \d+', sequence_content, re.IGNORECASE))
                    logger.info(f"[EMAIL-GENERATION] SUCCESS - Generated sequence with {email_count} emails, {len(sequence_content)} characters")
                    print(f"Successfully generated 9-email sequence ({len(sequence_content)} characters, {email_count} emails detected)")
                    
                    # Log if email count is not exactly 9
                    if email_count != 9:
                        logger.warning(f"[EMAIL-VALIDATION] Expected 9 emails, detected {email_count}")
                        print(f"  → Warning: Expected 9 emails, detected {email_count}")
                    else:
                        logger.info(f"[EMAIL-VALIDATION] Correct email count: {email_count}")
                        print(f"  → Confirmed: {email_count} emails generated")
                    
                    return sequence_content
                else:
                    raise Exception("Empty response from OpenAI for email sequence generation")

            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"[EMAIL-ERROR] Attempt {attempt + 1} failed: {str(e)}")

                # Enhanced error handling for email sequence context
                if "model" in error_msg and ("not found" in error_msg or "unavailable" in error_msg or "sunset" in error_msg):
                    logger.critical(f"[EMAIL-CRITICAL] OpenAI model {openai_model} unavailable for email sequence generation")
                    raise Exception(f"OpenAI model {openai_model} is unavailable or has been sunset. Please update the email sequence model configuration.")

                # Rate limiting specific handling
                if "rate" in error_msg and "limit" in error_msg:
                    logger.warning(f"[EMAIL-RATE-LIMIT] Hit rate limit during email sequence generation")
                    if attempt < max_retries - 1:
                        extended_delay = retry_delay * 3
                        logger.info(f"[EMAIL-RETRY] Rate limit - waiting {extended_delay}s before retry")
                        time.sleep(extended_delay)
                        continue

                if attempt < max_retries - 1:
                    logger.info(f"[EMAIL-RETRY] Retrying email sequence generation in {retry_delay}s...")
                    print(f"OpenAI API error: {e}. Retrying email sequence in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"[EMAIL-FAILURE] Failed to generate email sequence after {max_retries} attempts")
                    print(f"Failed to generate email sequence after {max_retries} attempts: {e}")
                    raise

    except Exception as e:
        print(f"Error in generate_email_sequence: {str(e)}")
        raise


# =====================================================================
# MAIN WORKFLOW FUNCTIONS
# =====================================================================

def load_guidance_files(bucket_name):
    """
    Load the style guide and example sequence from Supabase bucket
    
    Args:
        bucket_name (str): Name of the guidance bucket
        
    Returns:
        dict: Dictionary containing style guide and example sequence
    """
    print("\n" + "=" * 80)
    print("LOADING EMAIL SEQUENCE GUIDANCE FILES")
    print("=" * 80)

    try:
        # Load style guide and example sequence
        style_guide = download_file_from_bucket(bucket_name, "B2B Software Email Sequence Style Guide.txt")
        example_sequence = download_file_from_bucket(bucket_name, "Example Company Software 9-Email Sequence.txt")

        # Validate that guidance files have content
        if not style_guide or not style_guide.strip():
            raise Exception("B2B Software Email Sequence Style Guide.txt is empty or contains only whitespace")
        
        if not example_sequence or not example_sequence.strip():
            raise Exception("Example Company Software 9-Email Sequence.txt is empty or contains only whitespace")

        guidance_files = {
            "style_guide": style_guide,
            "example_sequence": example_sequence
        }

        print("Successfully loaded all email sequence guidance files")
        return guidance_files

    except Exception as e:
        print(f"Error loading guidance files: {str(e)}")
        raise Exception(f"Failed to load required guidance files from bucket '{bucket_name}': {str(e)}")


def determine_output_bucket(card_number, bucket_config):
    """
    Determine which output bucket to use based on card number
    
    Args:
        card_number (int): Card number (1-15)
        bucket_config (dict): Bucket configuration from Supabase
        
    Returns:
        str: Bucket name ('four-prob-emails' or 'two-prob-emails')
    """
    if 1 <= card_number <= 10:
        return bucket_config["output_four_prob"]
    elif 11 <= card_number <= 15:
        return bucket_config["output_two_prob"]
    else:
        raise Exception(f"Invalid card number: {card_number}. Must be between 1-15.")


def process_psq_cards(variables, guidance_files):
    """
    Process all 15 PSQ Card files sequentially
    
    Args:
        variables (dict): Configuration variables from Supabase
        guidance_files (dict): Loaded guidance files
        
    Returns:
        dict: Summary of processed email sequences
    """
    print("\n" + "=" * 80)
    print("PROCESSING PSQ CARDS FOR EMAIL SEQUENCES")
    print("=" * 80)

    try:
        # Extract configuration
        global_config = variables["global"]
        email_config = variables["scripts"]["email_sequence"]

        company_name = global_config.get("COMPANY_NAME", "company")
        card_combinations = email_config.get("card_combinations", [])
        bucket_config = email_config["supabase_buckets"]
        input_bucket = bucket_config["input_cards"]
        openai_model = email_config.get("openai_model", "gpt-4o")

        # Generate timestamp for output files
        timestamp = datetime.now(eastern_tz).strftime("%Y%m%d_%H%M")

        processed_sequences = []
        total_cards = 15  # Processing all 15 PSQ cards

        print(f"Processing {total_cards} PSQ Card files for {company_name}")
        print(f"Using OpenAI model: {openai_model}")
        print(f"Input bucket: {input_bucket}")

        # Process each PSQ card (1-15)
        for card_num in range(1, 16):
            card_progress = f"[{card_num}/{total_cards}]"
            logger.info(f"[EMAIL-PROGRESS] {card_progress} Starting processing for card {card_num}")
            print(f"\nProcessing email sequence for card {card_num} of {total_cards}...")

            try:
                # Get card combinations from configuration
                if card_num - 1 >= len(card_combinations):
                    raise Exception(f"Card combination not found in configuration for card {card_num}")
                
                problem_string = card_combinations[card_num - 1]
                
                # Construct input filename using company name + card number + problem string
                input_filename = f"{company_name}_card{card_num:02d}_{problem_string}.txt"

                print(f"Looking for file: {input_filename}")

                # Download the PSQ Card content (this will throw error if file is missing)
                try:
                    psq_card_content = download_file_from_bucket(input_bucket, input_filename)
                except Exception as e:
                    raise Exception(f"PSQ card file '{input_filename}' is missing from bucket '{input_bucket}': {str(e)}")

                # Extract problem string from filename for output naming (we already have it)
                # problem_string is already set above

                # Determine output bucket based on card number
                output_bucket = determine_output_bucket(card_num, bucket_config)

                # Generate output filename
                output_filename = f"{company_name}_email_seq_{card_num}_{problem_string}_{timestamp}.txt"

                logger.info(f"[EMAIL-PROGRESS] {card_progress} Input: {input_filename}")
                logger.info(f"[EMAIL-PROGRESS] {card_progress} Output: {output_filename} → {output_bucket}")
                logger.info(f"[EMAIL-PROGRESS] {card_progress} Problem string: {problem_string}")

                # Generate the email sequence
                logger.info(f"[EMAIL-PROGRESS] {card_progress} Generating 9-email sequence for {problem_string}")
                sequence_content = generate_email_sequence(
                    guidance_files["style_guide"],
                    guidance_files["example_sequence"], 
                    psq_card_content, 
                    openai_model)

                # Count emails in the generated sequence and validate
                email_count = len(re.findall(r'Email \d+', sequence_content, re.IGNORECASE))
                
                # Validate that exactly 9 emails were generated
                if email_count != 9:
                    logger.warning(f"[EMAIL-VALIDATION] Expected 9 emails, got {email_count} for card {card_num}")
                    print(f"  → Warning: Generated {email_count} emails instead of 9 for card {card_num}")
                    # Continue processing but log the issue
                else:
                    logger.info(f"[EMAIL-VALIDATION] Correct email count: {email_count} for card {card_num}")

                # Upload the generated sequence to appropriate bucket
                upload_file_to_bucket(output_bucket, output_filename, sequence_content)

                # Record the processed sequence
                processed_sequences.append({
                    "card_number": card_num,
                    "input_file": input_filename,
                    "output_file": output_filename,
                    "output_bucket": output_bucket,
                    "problem_string": problem_string,
                    "sequence_length": len(sequence_content),
                    "email_count": email_count,
                    "status": "success",
                    "sequence_type": "EMAIL"
                })

                logger.info(f"[EMAIL-SUCCESS] {card_progress} Generated {email_count} emails for {problem_string}")
                print(f"Successfully processed email sequence for card {card_num} ({problem_string})")

            except Exception as e:
                logger.error(f"[EMAIL-FAILURE] {card_progress} Failed processing card {card_num}: {str(e)}")
                print(f"Error processing card {card_num}: {str(e)}")
                processed_sequences.append({
                    "card_number": card_num,
                    "input_file": input_filename if 'input_filename' in locals() else "unknown",
                    "output_file": output_filename if 'output_filename' in locals() else "unknown",
                    "output_bucket": output_bucket if 'output_bucket' in locals() else "unknown",
                    "problem_string": problem_string if 'problem_string' in locals() else "unknown",
                    "error": str(e),
                    "status": "failed",
                    "sequence_type": "EMAIL"
                })

        # Generate summary with email count analysis
        successful_sequences = [s for s in processed_sequences if s["status"] == "success"]
        failed_sequences = [s for s in processed_sequences if s["status"] == "failed"]

        # Calculate email count statistics
        if successful_sequences:
            email_counts = [s["email_count"] for s in successful_sequences if "email_count" in s]
            avg_email_count = sum(email_counts) / len(email_counts) if email_counts else 0
            nine_email_count = len([s for s in successful_sequences if s.get("email_count") == 9])
        else:
            avg_email_count = 0
            nine_email_count = 0

        # Count by bucket
        four_prob_count = len([s for s in successful_sequences if s.get("output_bucket") == "four-prob-emails"])
        two_prob_count = len([s for s in successful_sequences if s.get("output_bucket") == "two-prob-emails"])

        summary = {
            "total_processed": total_cards,
            "successful": len(successful_sequences),
            "failed": len(failed_sequences),
            "sequences": processed_sequences,
            "company_name": company_name,
            "timestamp": timestamp,
            "openai_model": openai_model,
            "sequence_type": "EMAIL",
            "target_emails_per_sequence": 9,
            "bucket_distribution": {
                "four_prob_emails": four_prob_count,
                "two_prob_emails": two_prob_count
            },
            "email_count_analysis": {
                "average_email_count": round(avg_email_count, 1),
                "nine_email_compliance": f"{nine_email_count}/{len(successful_sequences)}" if successful_sequences else "0/0",
                "target": "9 emails per sequence"
            }
        }

        print(f"\nEmail sequence processing complete: {len(successful_sequences)}/{total_cards} sequences generated successfully")
        
        # Report email count compliance
        if successful_sequences:
            avg_emails = summary["email_count_analysis"]["average_email_count"]
            nine_ratio = summary["email_count_analysis"]["nine_email_compliance"]
            print(f"Email count analysis: Average {avg_emails} emails per sequence")
            print(f"9-email target compliance: {nine_ratio} sequences")
            print(f"Bucket distribution: {four_prob_count} four-problem, {two_prob_count} two-problem")

        return summary

    except Exception as e:
        print(f"Error in process_psq_cards: {str(e)}")
        raise


def main():
    """Main function to orchestrate the entire email sequence workflow."""
    try:
        logger.info("=" * 80)
        logger.info("EMAIL SEQUENCE AUTOMATION WORKFLOW - MAIN START")
        logger.info("=" * 80)
        print("=" * 80)
        print("EMAIL SEQUENCE AUTOMATION WORKFLOW")
        print("=" * 80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Script directory: {SCRIPT_DIR}")
        print(f"Target: 9-email sequences from 15 PSQ cards")

        # Fetch configuration from Supabase
        logger.info("[EMAIL-CONFIG] Fetching configuration from Supabase...")
        print("\nFetching email sequence configuration from Supabase...")
        
        try:
            variables = fetch_configuration_from_supabase()
        except Exception as e:
            logger.critical(f"[EMAIL-CRITICAL] Failed to fetch configuration: {str(e)}")
            raise Exception(f"Failed to fetch email sequence configuration from Supabase: {str(e)}")

        # Validate configuration structure
        if "scripts" not in variables or "email_sequence" not in variables["scripts"]:
            logger.critical("[EMAIL-CRITICAL] Missing email_sequence configuration in Supabase")
            raise Exception("email_sequence configuration not found in Supabase config. Please ensure the configuration includes an 'email_sequence' section.")

        if "global" not in variables:
            logger.critical("[EMAIL-CRITICAL] Missing global configuration in Supabase")
            raise Exception("global configuration not found in Supabase config.")
        
        global_config = variables["global"]
        email_config = variables["scripts"]["email_sequence"]
        logger.info("[EMAIL-CONFIG] Successfully loaded email_sequence configuration")

        # Validate required configuration
        company_name = global_config.get("COMPANY_NAME")
        if not company_name:
            raise Exception("COMPANY_NAME not found in global configuration.")

        # Validate bucket configuration
        if "supabase_buckets" not in email_config:
            raise Exception("supabase_buckets not found in email_sequence configuration.")
        
        bucket_config = email_config["supabase_buckets"]
        required_buckets = ["input_cards", "guidance", "output_four_prob", "output_two_prob"]
        for bucket_name in required_buckets:
            if bucket_name not in bucket_config:
                raise Exception(f"{bucket_name} bucket not found in configuration.")

        # Validate card combinations
        if "card_combinations" not in email_config:
            raise Exception("card_combinations not found in email_sequence configuration.")
        
        card_combinations = email_config["card_combinations"]
        if len(card_combinations) != 15:
            raise Exception(f"Expected exactly 15 card combinations, found {len(card_combinations)}. Please ensure all 15 PSQ card combinations are configured.")

        # Validate card combinations
        if "card_combinations" not in email_config:
            raise Exception("card_combinations not found in email_sequence configuration.")
        
        card_combinations = email_config["card_combinations"]
        if len(card_combinations) != 15:
            raise Exception(f"Expected exactly 15 card combinations, found {len(card_combinations)}. Please ensure all 15 PSQ card combinations are configured.")

        # Validate OpenAI model
        openai_model = email_config.get("openai_model", "gpt-4o")
        if not openai_model:
            raise Exception("openai_model not found in email_sequence configuration.")

        # Load guidance files
        try:
            guidance_files = load_guidance_files(bucket_config["guidance"])
            logger.info("[EMAIL-CONFIG] Successfully loaded guidance files")
        except Exception as e:
            logger.critical(f"[EMAIL-CRITICAL] Failed to load guidance files: {str(e)}")
            raise

        # Process PSQ Cards
        try:
            logger.info("[EMAIL-PROCESSING] Starting PSQ card processing for email sequences")
            summary = process_psq_cards(variables, guidance_files)
            logger.info(f"[EMAIL-PROCESSING] Completed processing: {summary['successful']}/{summary['total_processed']} successful")
        except Exception as e:
            logger.critical(f"[EMAIL-CRITICAL] Failed during PSQ card processing: {str(e)}")
            raise

        # Calculate and display execution time
        end_time = datetime.now(eastern_tz)
        execution_time = end_time - start_time

        logger.info("=" * 80)
        logger.info("EMAIL SEQUENCE WORKFLOW COMPLETE")
        logger.info(f"Execution time: {execution_time}")
        logger.info(f"Success rate: {summary['successful']}/{summary['total_processed']}")
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("EMAIL SEQUENCE WORKFLOW COMPLETE")
        print("=" * 80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {execution_time}")
        print(f"Email sequences generated: {summary['successful']}/{summary['total_processed']}")
        print(f"Summary saved as: {summary_filename}")

        if summary['failed'] > 0:
            logger.warning(f"[EMAIL-WARNING] {summary['failed']} sequences failed to generate")
            print(f"\nWarning: {summary['failed']} email sequence(s) failed to generate")
            for sequence in summary['sequences']:
                if sequence['status'] == 'failed':
                    logger.error(f"[EMAIL-FAILED-SEQ] Card {sequence['card_number']}: {sequence.get('error', 'Unknown error')}")
                    print(f"  - Card {sequence['card_number']}: {sequence.get('error', 'Unknown error')}")

        print("\nEmail sequence automation workflow completed successfully!")
        
        # Log successful session completion
        logger.info("=" * 60)
        logger.info("EMAIL SEQUENCE AUTOMATION - SESSION END (SUCCESS)")
        logger.info("=" * 60)

    except Exception as e:
        logger.critical(f"[EMAIL-CRITICAL] Critical error in email sequence workflow: {str(e)}")
        logger.critical(f"[EMAIL-CRITICAL] Traceback: {traceback.format_exc()}")
        print(f"\nCritical error in email sequence workflow: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Log session end on failure
        logger.info("=" * 60)
        logger.info("EMAIL SEQUENCE AUTOMATION - SESSION END (FAILED)")
        logger.info("=" * 60)
        raise


if __name__ == "__main__":
    main()