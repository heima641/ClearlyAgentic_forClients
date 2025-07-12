#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threads & Titles Automation Workflow Script

This script processes Poppy Card content and generates Twitter threads and video titles using:
- World-class threads guidance (examples and best practices)
- Threads prompt (specific generation instructions) 
- World-class titles guidance (title structure and patterns)
- Titles prompt (specific title generation directions)
- Poppy Card content (unique subject matter from all 15 cards)

SEQUENTIAL WORKFLOW:
1. THREADS GENERATION: Process all 15 PSQ cards → 15 individual Twitter thread files
2. SUCCESS VALIDATION: Ensure threads completed before proceeding  
3. TITLES GENERATION: Process all 15 PSQ cards → 1 consolidated file with 45 titles (3 per card)

Card Range: 01-15 (all available PSQ cards)
Output Format: Individual thread files + consolidated titles file
Processing: Sequential (threads → validation → titles)

All output files are saved to separate Supabase Storage buckets.
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

# Import OpenAI for content generation
import openai

# SUPABASE BUCKET INTEGRATION
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

# MODIFIED: Configure logging with THREADS-TITLES identifier instead of SHORT-SCRIPT
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [THREADS-TITLES] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"threads_titles_log_{datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}.log")
    ])
logger = logging.getLogger(__name__)

# MODIFIED: Log initialization for THREADS-TITLES workflow instead of SHORT workflow
logger.info("=" * 60)
logger.info("THREADS & TITLES AUTOMATION - SESSION START")
logger.info(f"Target: Twitter threads + Video titles from cards 01-15")
logger.info(f"Processing: Sequential (threads → validation → titles)")
logger.info(f"Session ID: {datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}")
logger.info("=" * 60)

# Suppress excessive HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Setup directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_PATH = os.path.join(script_dir, "STATIC_VARS_MAR2025.env")

# =====================================================================
# COMMON FUNCTIONS (ADAPTED FROM VIDEO_SCRIPT_SHORT)
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
        # MODIFIED: Updated logging identifier for threads-titles workflow
        logger.info(f"[THREADS-TITLES] Downloading {file_name} from {bucket_name} bucket...")
        print(f"Downloading {file_name} from {bucket_name} bucket...")
        response = supabase.storage.from_(bucket_name).download(file_name)

        if response:
            content = response.decode('utf-8')
            logger.info(f"[THREADS-TITLES] Successfully downloaded {file_name} ({len(content)} characters)")
            print(
                f"Successfully downloaded {file_name} ({len(content)} characters)"
            )
            return content
        else:
            raise Exception(f"Failed to download {file_name}")

    except Exception as e:
        logger.error(f"[THREADS-TITLES] Error downloading {file_name} from {bucket_name}: {str(e)}")
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
        # MODIFIED: Updated logging identifier for threads-titles workflow
        logger.info(f"[THREADS-TITLES] Uploading {file_name} to {bucket_name} bucket...")
        print(f"Uploading {file_name} to {bucket_name} bucket...")

        # Convert string content to bytes
        file_bytes = file_content.encode('utf-8')

        response = supabase.storage.from_(bucket_name).upload(
            file_name, file_bytes, {"content-type": "text/plain"})

        logger.info(f"[THREADS-TITLES] Successfully uploaded {file_name} to {bucket_name}")
        print(f"Successfully uploaded {file_name} to {bucket_name}")
        return True

    except Exception as e:
        logger.error(f"[THREADS-TITLES] Error uploading {file_name} to {bucket_name}: {str(e)}")
        print(f"Error uploading {file_name} to {bucket_name}: {str(e)}")
        raise


# =====================================================================
# NEW FUNCTIONS: THREAD AND TITLE GENERATION
# =====================================================================

def generate_twitter_thread(world_class_threads, threads_prompt, poppy_card_content, openai_model="gpt-4o", max_retries=3, retry_delay=2):
    """
    Generate a Twitter thread using OpenAI API with guidance and Poppy Card content
    
    Args:
        world_class_threads (str): Examples of high-quality Twitter threads
        threads_prompt (str): Specific instructions for thread generation
        poppy_card_content (str): Poppy Card content to base thread on
        openai_model (str): OpenAI model to use
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        str: Generated Twitter thread
    """
    try:
        # Create OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)

        # MODIFIED: Create specialized system prompt for Twitter thread generation
        system_prompt = f"""You are a professional social media content creator specializing in high-engagement Twitter threads. Generate a Twitter thread with these specifications:

WORLD-CLASS THREAD EXAMPLES AND PATTERNS:
{world_class_threads}

SPECIFIC THREAD GENERATION INSTRUCTIONS:
{threads_prompt}

CONTENT SOURCE (PSQ Card):
{poppy_card_content}

TWITTER THREAD REQUIREMENTS:
1. STRUCTURE: 8-12 tweets in thread format
2. OPENING: Hook tweet that stops the scroll
3. PROGRESSION: Logical flow with clear progression
4. ENGAGEMENT: Include questions, insights, and calls to action
5. FORMATTING: Use proper Twitter thread format with tweet numbers
6. HASHTAGS: Include 2-3 relevant hashtags strategically
7. CONCLUSION: Strong closing tweet with clear next step

THREAD OPTIMIZATION:
- Lead with the strongest insight or contradiction
- Use short, punchy sentences perfect for mobile reading  
- Include tweet-worthy quotes and key statistics
- Create anticipation for the next tweet
- End with engagement driver (question, poll idea, or CTA)

FORMAT EXAMPLE:
1/ [Hook tweet with strong opening]

2/ [First key insight or setup]

3/ [Supporting detail or example]

...

[X]/ [Strong conclusion with CTA]

Generate a Twitter thread that maximizes engagement while delivering real value from the PSQ card content."""

        # Attempt to generate thread with retry logic
        for attempt in range(max_retries):
            try:
                logger.info(f"[THREAD-GENERATION] Attempt {attempt + 1}/{max_retries} for Twitter thread")
                print(f"Generating Twitter thread using {openai_model} (attempt {attempt + 1}/{max_retries})...")

                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role": "user",
                        "content": "Generate a high-engagement Twitter thread now. Use the PSQ card content as your foundation and follow the world-class examples provided. Format as a proper Twitter thread with numbered tweets."
                    }],
                    max_tokens=2000,
                    temperature=0.7)

                thread_content = response.choices[0].message.content
                if thread_content:
                    thread_content = thread_content.strip()

                if thread_content:
                    # Count tweets in thread for validation
                    tweet_count = len([line for line in thread_content.split('\n') if re.match(r'^\d+/', line.strip())])
                    logger.info(f"[THREAD-GENERATION] SUCCESS - Generated {tweet_count} tweets, {len(thread_content)} characters")
                    print(f"Successfully generated Twitter thread ({tweet_count} tweets, {len(thread_content)} characters)")
                    return thread_content
                else:
                    raise Exception("Empty response from OpenAI for Twitter thread generation")

            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"[THREAD-ERROR] Attempt {attempt + 1} failed: {str(e)}")

                # Enhanced error handling for thread generation
                if "model" in error_msg and ("not found" in error_msg or "unavailable" in error_msg):
                    logger.critical(f"[THREAD-CRITICAL] OpenAI model {openai_model} unavailable for thread generation")
                    raise Exception(f"OpenAI model {openai_model} is unavailable. Please update the model configuration.")

                if "rate" in error_msg and "limit" in error_msg:
                    logger.warning(f"[THREAD-RATE-LIMIT] Hit rate limit during thread generation")
                    if attempt < max_retries - 1:
                        extended_delay = retry_delay * 3
                        logger.info(f"[THREAD-RETRY] Rate limit - waiting {extended_delay}s before retry")
                        time.sleep(extended_delay)
                        continue

                if attempt < max_retries - 1:
                    logger.info(f"[THREAD-RETRY] Retrying thread generation in {retry_delay}s...")
                    print(f"OpenAI API error: {e}. Retrying thread generation in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"[THREAD-FAILURE] Failed to generate thread after {max_retries} attempts")
                    print(f"Failed to generate thread after {max_retries} attempts: {e}")
                    raise

    except Exception as e:
        print(f"Error in generate_twitter_thread: {str(e)}")
        raise


def generate_video_titles(world_class_titles, titles_prompt, poppy_card_content, card_number, openai_model="gpt-4o", max_retries=3, retry_delay=2):
    """
    Generate 3 video titles for a specific PSQ card
    
    Args:
        world_class_titles (str): Examples of high-quality video titles
        titles_prompt (str): Specific instructions for title generation
        poppy_card_content (str): Poppy Card content to base titles on
        card_number (str): Card identifier for labeling
        openai_model (str): OpenAI model to use
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        str: Generated titles formatted for consolidation (3 titles with card label)
    """
    try:
        # Create OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)

        # MODIFIED: Create specialized system prompt for video title generation
        system_prompt = f"""You are a professional video marketing expert specializing in high-converting YouTube titles. Generate video titles with these specifications:

WORLD-CLASS TITLE EXAMPLES AND PATTERNS:
{world_class_titles}

SPECIFIC TITLE GENERATION INSTRUCTIONS:
{titles_prompt}

CONTENT SOURCE (PSQ Card):
{poppy_card_content}

VIDEO TITLE REQUIREMENTS:
1. QUANTITY: Generate exactly 3 distinct title options
2. LENGTH: 60-70 characters optimal for YouTube display
3. ENGAGEMENT: Use power words, numbers, and curiosity gaps
4. CLARITY: Immediately communicate the video's value proposition
5. VARIETY: Each title should take a different angle on the same content
6. SEO: Include relevant keywords naturally
7. AVOID: Clickbait without substance, ALL CAPS, excessive emojis

TITLE OPTIMIZATION STRATEGIES:
- Lead with numbers, secrets, or "how to" when appropriate
- Use emotional triggers (fear, curiosity, desire, urgency)
- Include specific benefits or outcomes
- Test different formulas (listicle, question, bold statement)
- Make each title distinctive in approach but consistent in value

FORMAT REQUIREMENT:
Return exactly 3 titles, each on a separate line, numbered 1-3.

Example output format:
1. [First title option]
2. [Second title option] 
3. [Third title option]

Generate 3 video titles that would maximize click-through rates while delivering on the promise."""

        # Attempt to generate titles with retry logic
        for attempt in range(max_retries):
            try:
                logger.info(f"[TITLES-GENERATION] Attempt {attempt + 1}/{max_retries} for video titles - {card_number}")
                print(f"Generating video titles for {card_number} using {openai_model} (attempt {attempt + 1}/{max_retries})...")

                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role": "user",
                        "content": f"Generate 3 high-converting video titles for {card_number}. Use the PSQ card content as your foundation and follow the world-class title patterns provided. Return exactly 3 numbered titles."
                    }],
                    max_tokens=500,
                    temperature=0.8)

                titles_content = response.choices[0].message.content
                if titles_content:
                    titles_content = titles_content.strip()

                if titles_content:
                    # Format titles with card label for consolidation
                    formatted_titles = f"=== {card_number.upper()} TITLES ===\n{titles_content}\n"
                    
                    # Count generated titles for validation
                    title_count = len([line for line in titles_content.split('\n') if re.match(r'^\d+\.', line.strip())])
                    logger.info(f"[TITLES-GENERATION] SUCCESS - Generated {title_count} titles for {card_number}")
                    print(f"Successfully generated {title_count} video titles for {card_number}")
                    return formatted_titles
                else:
                    raise Exception(f"Empty response from OpenAI for video titles generation - {card_number}")

            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"[TITLES-ERROR] Attempt {attempt + 1} failed for {card_number}: {str(e)}")

                # Enhanced error handling for titles generation
                if "model" in error_msg and ("not found" in error_msg or "unavailable" in error_msg):
                    logger.critical(f"[TITLES-CRITICAL] OpenAI model {openai_model} unavailable for titles generation")
                    raise Exception(f"OpenAI model {openai_model} is unavailable. Please update the model configuration.")

                if "rate" in error_msg and "limit" in error_msg:
                    logger.warning(f"[TITLES-RATE-LIMIT] Hit rate limit during titles generation")
                    if attempt < max_retries - 1:
                        extended_delay = retry_delay * 3
                        logger.info(f"[TITLES-RETRY] Rate limit - waiting {extended_delay}s before retry")
                        time.sleep(extended_delay)
                        continue

                if attempt < max_retries - 1:
                    logger.info(f"[TITLES-RETRY] Retrying titles generation for {card_number} in {retry_delay}s...")
                    print(f"OpenAI API error: {e}. Retrying titles generation in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"[TITLES-FAILURE] Failed to generate titles for {card_number} after {max_retries} attempts")
                    print(f"Failed to generate titles for {card_number} after {max_retries} attempts: {e}")
                    raise

    except Exception as e:
        print(f"Error in generate_video_titles: {str(e)}")
        raise


# =====================================================================
# MAIN WORKFLOW FUNCTIONS (MODIFIED FROM VIDEO_SCRIPT_SHORT)
# =====================================================================

def load_guidance_files(bucket_name):
    """
    MODIFIED: Load the four guidance files from Supabase bucket (threads & titles version)
    
    Args:
        bucket_name (str): Name of the guidance-threads-titles bucket
        
    Returns:
        dict: Dictionary containing all four guidance files
    """
    print("\n" + "=" * 80)
    print("LOADING GUIDANCE FILES FOR THREADS & TITLES")
    print("=" * 80)

    try:
        # MODIFIED: Load four guidance files instead of three (threads and titles specific)
        world_class_threads = download_file_from_bucket(bucket_name, "world-class-threads.txt")
        threads_prompt = download_file_from_bucket(bucket_name, "threads-prompt.txt")
        world_class_titles = download_file_from_bucket(bucket_name, "world-class-titles.txt")
        titles_prompt = download_file_from_bucket(bucket_name, "titles-prompt.txt")

        # Validate that all guidance files have content
        guidance_files = {
            "world_class_threads": world_class_threads,
            "threads_prompt": threads_prompt,
            "world_class_titles": world_class_titles,
            "titles_prompt": titles_prompt
        }

        for file_key, content in guidance_files.items():
            if not content or not content.strip():
                raise Exception(f"{file_key}.txt is empty or contains only whitespace")

        print("Successfully loaded all guidance files for threads & titles")
        return guidance_files

    except Exception as e:
        print(f"Error loading guidance files: {str(e)}")
        raise Exception(f"Failed to load required guidance files from bucket '{bucket_name}': {str(e)}")


def process_threads_workflow(company_name, card_combinations, input_bucket, output_bucket, guidance_files, openai_model, timestamp):
    """
    MODIFIED: Process all 15 PSQ cards to generate individual Twitter thread files
    
    Args:
        company_name (str): Company name for file naming
        card_combinations (list): List of card combinations to process
        input_bucket (str): Input bucket name
        output_bucket (str): Output bucket name for threads
        guidance_files (dict): Loaded guidance files
        openai_model (str): OpenAI model name
        timestamp (str): Timestamp for file naming
        
    Returns:
        dict: Summary of processed threads
    """
    print("\n" + "=" * 80)
    print("PHASE 1: PROCESSING TWITTER THREADS (CARDS 01-15)")
    print("=" * 80)

    try:
        processed_threads = []
        total_cards = len(card_combinations)

        print(f"Processing {total_cards} Twitter threads for {company_name}")
        print(f"Card range: 01-15 (all PSQ cards)")
        print(f"Using OpenAI model: {openai_model}")
        print(f"Input bucket: {input_bucket}")
        print(f"Output bucket: {output_bucket}")

        # MODIFIED: Process all 15 cards (01-15) instead of 5 cards (11-15)
        for i, combination in enumerate(card_combinations, 1):
            card_progress = f"[{i}/{total_cards}]"
            card_number = f"card{i:02d}"  # Formats as card01, card02, ..., card15
            
            logger.info(f"[THREADS-PROGRESS] {card_progress} Starting thread generation for {card_number} - {combination}")
            print(f"\nProcessing Twitter thread {i} of {total_cards}...")
            print(f"Card: {card_number}, Combination: {combination}")

            try:
                # MODIFIED: Use standard card numbering (01-15) and thread file naming
                input_filename = f"{company_name}_{card_number}_{combination}.txt"
                output_filename = f"{company_name}_thread_{card_number}_{timestamp}.txt"

                logger.info(f"[THREADS-PROGRESS] {card_progress} Input: {input_filename}")
                logger.info(f"[THREADS-PROGRESS] {card_progress} Output: {output_filename}")
                print(f"Looking for file: {input_filename}")

                # Download the Poppy Card content
                poppy_card_content = download_file_from_bucket(input_bucket, input_filename)

                # MODIFIED: Generate Twitter thread instead of video script
                logger.info(f"[THREADS-PROGRESS] {card_progress} Generating Twitter thread for {combination}")
                thread_content = generate_twitter_thread(
                    guidance_files["world_class_threads"], 
                    guidance_files["threads_prompt"],
                    poppy_card_content, 
                    openai_model
                )

                # Count tweets for validation
                tweet_count = len([line for line in thread_content.split('\n') if re.match(r'^\d+/', line.strip())])

                # Upload the generated thread
                upload_file_to_bucket(output_bucket, output_filename, thread_content)

                # Record the processed thread
                processed_threads.append({
                    "card_number": card_number,
                    "combination": combination,
                    "input_file": input_filename,
                    "output_file": output_filename,
                    "content_length": len(thread_content),
                    "tweet_count": tweet_count,
                    "status": "success",
                    "content_type": "twitter_thread"
                })

                logger.info(f"[THREADS-SUCCESS] {card_progress} Generated {tweet_count} tweets for {card_number}")
                print(f"Successfully processed thread for {card_number}")

            except Exception as e:
                logger.error(f"[THREADS-FAILURE] {card_progress} Failed processing {card_number}: {str(e)}")
                print(f"Error processing thread for {card_number}: {str(e)}")
                processed_threads.append({
                    "card_number": card_number,
                    "combination": combination,
                    "input_file": input_filename if 'input_filename' in locals() else "unknown",
                    "output_file": output_filename if 'output_filename' in locals() else "unknown",
                    "error": str(e),
                    "status": "failed",
                    "content_type": "twitter_thread"
                })

        # Generate threads summary
        successful_threads = [t for t in processed_threads if t["status"] == "success"]
        failed_threads = [t for t in processed_threads if t["status"] == "failed"]

        threads_summary = {
            "total_processed": total_cards,
            "successful": len(successful_threads),
            "failed": len(failed_threads),
            "threads": processed_threads,
            "average_tweets": round(sum(t.get("tweet_count", 0) for t in successful_threads) / len(successful_threads), 1) if successful_threads else 0
        }

        print(f"\nTwitter threads processing complete: {len(successful_threads)}/{total_cards} threads generated successfully")
        if successful_threads:
            avg_tweets = threads_summary["average_tweets"]
            print(f"Average tweets per thread: {avg_tweets}")

        return threads_summary

    except Exception as e:
        print(f"Error in process_threads_workflow: {str(e)}")
        raise


def process_titles_workflow(company_name, card_combinations, input_bucket, output_bucket, guidance_files, openai_model, timestamp):
    """
    MODIFIED: Process all 15 PSQ cards to generate consolidated video titles file (3 titles per card)
    
    Args:
        company_name (str): Company name for file naming
        card_combinations (list): List of card combinations to process
        input_bucket (str): Input bucket name
        output_bucket (str): Output bucket name for titles
        guidance_files (dict): Loaded guidance files
        openai_model (str): OpenAI model name
        timestamp (str): Timestamp for file naming
        
    Returns:
        dict: Summary of processed titles
    """
    print("\n" + "=" * 80)
    print("PHASE 2: PROCESSING VIDEO TITLES (CARDS 01-15)")
    print("=" * 80)

    try:
        all_titles_content = []
        processed_titles = []
        total_cards = len(card_combinations)

        print(f"Processing {total_cards * 3} video titles for {company_name}")
        print(f"Format: 3 titles per PSQ card, consolidated into single file")
        print(f"Card range: 01-15 (all PSQ cards)")
        print(f"Using OpenAI model: {openai_model}")
        print(f"Input bucket: {input_bucket}")
        print(f"Output bucket: {output_bucket}")

        # Add header to consolidated file
        file_header = f"""# VIDEO TITLES GENERATED FROM PSQ CARDS
# Company: {company_name}
# Generated: {datetime.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S')} Eastern
# Total Cards Processed: {total_cards}
# Expected Titles: {total_cards * 3} (3 per card)
# 
# FORMAT: Each card section contains 3 title options
# ================================================

"""
        all_titles_content.append(file_header)

        # MODIFIED: Process all 15 cards to generate 3 titles each
        for i, combination in enumerate(card_combinations, 1):
            card_progress = f"[{i}/{total_cards}]"
            card_number = f"card{i:02d}"  # Formats as card01, card02, ..., card15
            
            logger.info(f"[TITLES-PROGRESS] {card_progress} Starting titles generation for {card_number} - {combination}")
            print(f"\nProcessing video titles {i} of {total_cards}...")
            print(f"Card: {card_number}, Combination: {combination}")

            try:
                # Use same input files as threads workflow
                input_filename = f"{company_name}_{card_number}_{combination}.txt"

                logger.info(f"[TITLES-PROGRESS] {card_progress} Input: {input_filename}")
                print(f"Looking for file: {input_filename}")

                # Download the Poppy Card content
                poppy_card_content = download_file_from_bucket(input_bucket, input_filename)

                # MODIFIED: Generate 3 video titles instead of video script or thread
                logger.info(f"[TITLES-PROGRESS] {card_progress} Generating 3 video titles for {card_number}")
                titles_content = generate_video_titles(
                    guidance_files["world_class_titles"], 
                    guidance_files["titles_prompt"],
                    poppy_card_content, 
                    card_number,
                    openai_model
                )

                # Add to consolidated content
                all_titles_content.append(titles_content)
                all_titles_content.append("\n")  # Spacing between card sections

                # Count generated titles for validation
                title_count = len([line for line in titles_content.split('\n') if re.match(r'^\d+\.', line.strip())])

                # Record the processed titles
                processed_titles.append({
                    "card_number": card_number,
                    "combination": combination,
                    "input_file": input_filename,
                    "title_count": title_count,
                    "content_length": len(titles_content),
                    "status": "success",
                    "content_type": "video_titles"
                })

                logger.info(f"[TITLES-SUCCESS] {card_progress} Generated {title_count} titles for {card_number}")
                print(f"Successfully processed {title_count} titles for {card_number}")

            except Exception as e:
                logger.error(f"[TITLES-FAILURE] {card_progress} Failed processing {card_number}: {str(e)}")
                print(f"Error processing titles for {card_number}: {str(e)}")
                
                # Add error section to consolidated file
                error_section = f"=== {card_number.upper()} TITLES ===\nERROR: Failed to generate titles - {str(e)}\n\n"
                all_titles_content.append(error_section)
                
                processed_titles.append({
                    "card_number": card_number,
                    "combination": combination,
                    "input_file": input_filename if 'input_filename' in locals() else "unknown",
                    "error": str(e),
                    "status": "failed",
                    "content_type": "video_titles"
                })

        # MODIFIED: Create single consolidated titles file instead of individual files
        consolidated_filename = f"{company_name}_titles_consolidated_{timestamp}.txt"
        consolidated_content = "".join(all_titles_content)
        
        # Upload the consolidated titles file
        upload_file_to_bucket(output_bucket, consolidated_filename, consolidated_content)

        # Generate titles summary
        successful_titles = [t for t in processed_titles if t["status"] == "success"]
        failed_titles = [t for t in processed_titles if t["status"] == "failed"]
        total_titles_generated = sum(t.get("title_count", 0) for t in successful_titles)

        titles_summary = {
            "total_cards_processed": total_cards,
            "successful_cards": len(successful_titles),
            "failed_cards": len(failed_titles),
            "total_titles_generated": total_titles_generated,
            "expected_titles": total_cards * 3,
            "consolidated_file": consolidated_filename,
            "titles": processed_titles
        }

        print(f"\nVideo titles processing complete: {len(successful_titles)}/{total_cards} cards processed successfully")
        print(f"Total titles generated: {total_titles_generated}/{total_cards * 3}")
        print(f"Consolidated file: {consolidated_filename}")

        return titles_summary

    except Exception as e:
        print(f"Error in process_titles_workflow: {str(e)}")
        raise


def validate_threads_success(threads_summary):
    """
    NEW: Validate that threads workflow completed successfully before proceeding to titles
    
    Args:
        threads_summary (dict): Summary from threads processing
        
    Returns:
        bool: True if validation passes
        
    Raises:
        Exception: If validation fails
    """
    try:
        logger.info("[VALIDATION] Validating threads workflow completion...")
        print("\n" + "=" * 80)
        print("VALIDATION: CHECKING THREADS WORKFLOW SUCCESS")
        print("=" * 80)

        total_processed = threads_summary["total_processed"]
        successful = threads_summary["successful"]
        failed = threads_summary["failed"]

        print(f"Threads processed: {successful}/{total_processed}")
        print(f"Failed threads: {failed}")

        # Define success criteria
        success_rate = successful / total_processed if total_processed > 0 else 0
        minimum_success_rate = 0.8  # Require 80% success rate

        if success_rate >= minimum_success_rate:
            logger.info(f"[VALIDATION] SUCCESS - Threads validation passed ({success_rate:.1%} success rate)")
            print(f"✅ Validation PASSED: {success_rate:.1%} success rate (minimum: {minimum_success_rate:.1%})")
            print("Proceeding to titles generation...")
            return True
        else:
            logger.error(f"[VALIDATION] FAILED - Insufficient success rate ({success_rate:.1%} < {minimum_success_rate:.1%})")
            print(f"❌ Validation FAILED: {success_rate:.1%} success rate below minimum {minimum_success_rate:.1%}")
            
            # Provide detailed failure analysis
            print("\nFailed threads analysis:")
            for thread in threads_summary["threads"]:
                if thread["status"] == "failed":
                    print(f"  - {thread['card_number']}: {thread.get('error', 'Unknown error')}")
            
            raise Exception(f"Threads workflow validation failed: {success_rate:.1%} success rate below required {minimum_success_rate:.1%}. Please check failed threads and retry.")

    except Exception as e:
        logger.error(f"[VALIDATION] Error during validation: {str(e)}")
        print(f"Error during threads validation: {str(e)}")
        raise


# =====================================================================
# MAIN FUNCTION (MODIFIED FROM VIDEO_SCRIPT_SHORT)
# =====================================================================

def main():
    """Main function to orchestrate the entire THREADS & TITLES workflow."""
    try:
        logger.info("=" * 80)
        logger.info("THREADS & TITLES AUTOMATION WORKFLOW - MAIN START")
        logger.info("=" * 80)
        print("=" * 80)
        print("THREADS & TITLES AUTOMATION WORKFLOW")
        print("=" * 80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Script directory: {SCRIPT_DIR}")
        print(f"Target: Twitter threads + Video titles from cards 01-15")
        print(f"Processing: Sequential (threads → validation → titles)")

        # MODIFIED: Fetch threads_titles configuration instead of video_script_short
        logger.info("[THREADS-TITLES-CONFIG] Fetching configuration from Supabase...")
        print("\nFetching THREADS & TITLES configuration from Supabase...")
        
        try:
            variables = fetch_configuration_from_supabase()
        except Exception as e:
            logger.critical(f"[THREADS-TITLES-CRITICAL] Failed to fetch configuration: {str(e)}")
            raise Exception(f"Failed to fetch threads & titles configuration from Supabase: {str(e)}")

        # MODIFIED: Validate that we have the threads_titles configuration instead of video_script_short
        if "scripts" not in variables or "threads_titles" not in variables["scripts"]:
            logger.critical("[THREADS-TITLES-CRITICAL] Missing threads_titles configuration in Supabase")
            raise Exception(
                "threads_titles configuration not found in Supabase config. Please ensure the configuration includes a 'threads_titles' section.")

        # Validate global configuration (same as original)
        if "global" not in variables:
            logger.critical("[THREADS-TITLES-CRITICAL] Missing global configuration in Supabase")
            raise Exception("global configuration not found in Supabase config. Please ensure the configuration includes a 'global' section.")
        
        global_config = variables["global"]
        if "COMPANY_NAME" not in global_config:
            logger.critical("[THREADS-TITLES-CRITICAL] Missing COMPANY_NAME in global configuration")
            raise Exception("COMPANY_NAME not found in global configuration. Please ensure the configuration includes a valid COMPANY_NAME.")
        
        company_name = global_config["COMPANY_NAME"]
        if not company_name or not company_name.strip():
            logger.critical("[THREADS-TITLES-CRITICAL] Empty COMPANY_NAME in configuration")
            raise Exception("COMPANY_NAME is empty or contains only whitespace. Please ensure the configuration includes a valid company name.")

        # MODIFIED: Load threads_titles configuration instead of video_script_short
        threads_titles_config = variables["scripts"]["threads_titles"]
        logger.info("[THREADS-TITLES-CONFIG] Successfully loaded threads_titles configuration")

        # MODIFIED: Validate required configuration structure for threads_titles
        if "supabase_buckets" not in threads_titles_config:
            logger.critical("[THREADS-TITLES-CRITICAL] Missing supabase_buckets in threads_titles configuration")
            raise Exception("supabase_buckets not found in threads_titles configuration. Please ensure the configuration includes bucket definitions.")
        
        bucket_config = threads_titles_config["supabase_buckets"]
        # MODIFIED: Check for threads_titles specific buckets
        required_buckets = ["input_cards", "guidance", "output_threads", "output_titles"]
        for bucket_name in required_buckets:
            if bucket_name not in bucket_config:
                logger.critical(f"[THREADS-TITLES-CRITICAL] Missing {bucket_name} bucket in configuration")
                raise Exception(f"{bucket_name} bucket not found in supabase_buckets configuration. Please ensure all required buckets are defined.")

        # Validate card combinations configuration (same validation but different context)
        if "card_combinations" not in threads_titles_config:
            logger.critical("[THREADS-TITLES-CRITICAL] Missing card_combinations in threads_titles configuration")
            raise Exception("card_combinations not found in threads_titles configuration. Please ensure the configuration includes card_combinations array.")
        
        card_combinations = threads_titles_config["card_combinations"]
        if not card_combinations:
            logger.critical("[THREADS-TITLES-CRITICAL] Empty card_combinations array in configuration")
            raise Exception("card_combinations array is empty. Please ensure the configuration includes card combinations for cards 01-15.")
        
        # MODIFIED: Expect 15 combinations for cards 01-15 instead of 5 for cards 11-15
        if len(card_combinations) != 15:
            logger.critical(f"[THREADS-TITLES-CRITICAL] Expected 15 card combinations, found {len(card_combinations)}")
            raise Exception(f"Expected exactly 15 card combinations for cards 01-15, found {len(card_combinations)}. Please ensure the configuration includes exactly 15 combinations.")

        # Validate OpenAI model configuration (same as original)
        if "openai_model" not in threads_titles_config:
            logger.critical("[THREADS-TITLES-CRITICAL] Missing openai_model in threads_titles configuration")
            raise Exception("openai_model not found in threads_titles configuration. Please ensure the configuration includes a valid openai_model.")
        
        openai_model = threads_titles_config["openai_model"]
        if not openai_model or not openai_model.strip():
            logger.critical("[THREADS-TITLES-CRITICAL] Empty openai_model in configuration")
            raise Exception("openai_model is empty or contains only whitespace. Please ensure the configuration includes a valid OpenAI model name.")

        # MODIFIED: Load guidance files for threads & titles (4 files instead of 3)
        if "guidance" not in bucket_config:
            logger.critical("[THREADS-TITLES-CRITICAL] Missing guidance bucket in configuration")
            raise Exception("guidance bucket not found in supabase_buckets configuration.")
        
        guidance_bucket = bucket_config["guidance"]
        try:
            guidance_files = load_guidance_files(guidance_bucket)
            logger.info("[THREADS-TITLES-CONFIG] Successfully loaded all guidance files")
        except Exception as e:
            logger.critical(f"[THREADS-TITLES-CRITICAL] Failed to load guidance files: {str(e)}")
            raise Exception(f"Failed to load threads & titles guidance files from bucket '{guidance_bucket}': {str(e)}")

        # Generate timestamp for both workflows
        timestamp = datetime.now(eastern_tz).strftime("%Y%m%d_%H%M")

        # MODIFIED: Execute sequential workflow instead of single processing
        # PHASE 1: Process Twitter Threads
        try:
            logger.info("[THREADS-TITLES-PROCESSING] Starting Phase 1: Twitter Threads")
            threads_summary = process_threads_workflow(
                company_name, 
                card_combinations, 
                bucket_config["input_cards"], 
                bucket_config["output_threads"], 
                guidance_files, 
                openai_model, 
                timestamp
            )
            logger.info(f"[THREADS-TITLES-PROCESSING] Phase 1 completed: {threads_summary['successful']}/{threads_summary['total_processed']} successful")
        except Exception as e:
            logger.critical(f"[THREADS-TITLES-CRITICAL] Failed during Phase 1 (threads): {str(e)}")
            raise Exception(f"Failed to process Twitter threads: {str(e)}")

        # VALIDATION: Check threads success before proceeding
        try:
            validate_threads_success(threads_summary)
        except Exception as e:
            logger.critical(f"[THREADS-TITLES-CRITICAL] Threads validation failed: {str(e)}")
            raise Exception(f"Threads workflow validation failed: {str(e)}")

        # PHASE 2: Process Video Titles (only if threads succeeded)
        try:
            logger.info("[THREADS-TITLES-PROCESSING] Starting Phase 2: Video Titles")
            titles_summary = process_titles_workflow(
                company_name, 
                card_combinations, 
                bucket_config["input_cards"], 
                bucket_config["output_titles"], 
                guidance_files, 
                openai_model, 
                timestamp
            )
            logger.info(f"[THREADS-TITLES-PROCESSING] Phase 2 completed: {titles_summary['successful_cards']}/{titles_summary['total_cards_processed']} successful")
        except Exception as e:
            logger.critical(f"[THREADS-TITLES-CRITICAL] Failed during Phase 2 (titles): {str(e)}")
            raise Exception(f"Failed to process video titles: {str(e)}")

        # MODIFIED: Create combined summary for both workflows
        combined_summary = {
            "timestamp": timestamp,
            "company_name": company_name,
            "openai_model": openai_model,
            "processing_sequence": "sequential",
            "threads_workflow": threads_summary,
            "titles_workflow": titles_summary,
            "overall_success": {
                "threads_success_rate": threads_summary['successful'] / threads_summary['total_processed'],
                "titles_success_rate": titles_summary['successful_cards'] / titles_summary['total_cards_processed'],
                "total_outputs": {
                    "thread_files": threads_summary['successful'],
                    "consolidated_titles_file": 1 if titles_summary['successful_cards'] > 0 else 0,
                    "total_titles_generated": titles_summary['total_titles_generated']
                }
            }
        }

        # MODIFIED: Save combined summary instead of single workflow summary
        summary_filename = f"threads_titles_summary_{timestamp}.json"
        summary_content = json.dumps(combined_summary, indent=2)
        
        try:
            # Save to threads bucket (could also be saved to a dedicated summary bucket)
            upload_file_to_bucket(bucket_config["output_threads"], summary_filename, summary_content)
            logger.info(f"[THREADS-TITLES-SUMMARY] Saved combined summary as {summary_filename}")
        except Exception as e:
            logger.error(f"[THREADS-TITLES-ERROR] Failed to save summary: {str(e)}")
            print(f"Warning: Could not save summary file: {str(e)}")
            print("Threads & titles generation completed successfully despite summary save failure.")

        # Calculate and display execution time
        end_time = datetime.now(eastern_tz)
        execution_time = end_time - start_time

        logger.info("=" * 80)
        logger.info("THREADS & TITLES WORKFLOW COMPLETE")
        logger.info(f"Execution time: {execution_time}")
        logger.info(f"Threads success: {threads_summary['successful']}/{threads_summary['total_processed']}")
        logger.info(f"Titles success: {titles_summary['successful_cards']}/{titles_summary['total_cards_processed']}")
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("THREADS & TITLES WORKFLOW COMPLETE")
        print("=" * 80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {execution_time}")
        print(f"Twitter threads generated: {threads_summary['successful']}/{threads_summary['total_processed']}")
        print(f"Video titles generated: {titles_summary['total_titles_generated']} titles from {titles_summary['successful_cards']} cards")
        print(f"Output files created:")
        print(f"  - {threads_summary['successful']} individual thread files")
        print(f"  - 1 consolidated titles file ({titles_summary['consolidated_file']})")
        print(f"Summary saved as: {summary_filename}")

        # Report any failures
        total_failures = threads_summary['failed'] + titles_summary['failed_cards']
        if total_failures > 0:
            logger.warning(f"[THREADS-TITLES-WARNING] {total_failures} total failures across both workflows")
            print(f"\nWarning: {total_failures} total failure(s) across both workflows")
            
            if threads_summary['failed'] > 0:
                print(f"Thread failures ({threads_summary['failed']}):")
                for thread in threads_summary['threads']:
                    if thread['status'] == 'failed':
                        logger.error(f"[THREADS-FAILED] {thread['card_number']}: {thread.get('error', 'Unknown error')}")
                        print(f"  - {thread['card_number']}: {thread.get('error', 'Unknown error')}")
            
            if titles_summary['failed_cards'] > 0:
                print(f"Titles failures ({titles_summary['failed_cards']}):")
                for title in titles_summary['titles']:
                    if title['status'] == 'failed':
                        logger.error(f"[TITLES-FAILED] {title['card_number']}: {title.get('error', 'Unknown error')}")
                        print(f"  - {title['card_number']}: {title.get('error', 'Unknown error')}")

        print("\nThreads & Titles automation workflow completed successfully!")
        
        # Log successful session completion
        logger.info("=" * 60)
        logger.info("THREADS & TITLES AUTOMATION - SESSION END (SUCCESS)")
        logger.info("=" * 60)

    except Exception as e:
        logger.critical(f"[THREADS-TITLES-CRITICAL] Critical error in threads & titles workflow: {str(e)}")
        logger.critical(f"[THREADS-TITLES-CRITICAL] Traceback: {traceback.format_exc()}")
        print(f"\nCritical error in threads & titles workflow: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Ensure we log session end even on failure
        logger.info("=" * 60)
        logger.info("THREADS & TITLES AUTOMATION - SESSION END (FAILED)")
        logger.info("=" * 60)
        raise


if __name__ == "__main__":
    main()
    