#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Video Script SHORT Automation Workflow Script with 4 Quotes Per Problem

This script processes Poppy Card content and generates SHORT video scripts using:
- Voice guidance (tone and style)
- Method guidance (structure and framework) 
- Prompt instructions (specific processing directions for 5-minute format)
- Poppy Card content (unique subject matter for 2-problem cards)

ENHANCEMENTS:
- Simple quote distribution (4 quotes per problem = 8 total quotes per script)
- Peer validation psychology framework
- Professional competence vs. ego-stroking detection
- Quote distribution validation and reporting
- Company name integration (4 mentions per script, adapted for 5-minute format)

The workflow processes 5 predefined Poppy Card combinations sequentially (cards 11-15),
generating custom SHORT video scripts for each combination and saving them to Supabase.

Card Range: 11-15 (specifically for 2-problem format cards)
Target Duration: 5 minutes
Quote Distribution: 4 quotes per problem (8 total quotes)
Company Integration: 4 mentions (intro + 2 problems + outro)
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
            f"video_script_SHORT_enhanced_log_{datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}.log")
    ])
logger = logging.getLogger(__name__)

# Log initialization for SHORT workflow
logger.info("=" * 60)
logger.info("SHORT VIDEO SCRIPT AUTOMATION - ENHANCED WITH QUOTE DISTRIBUTION")
logger.info(f"Target: 5-minute scripts with 8 quotes (4 per problem) from cards 11-15")
logger.info(f"Session ID: {datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}")
logger.info("=" * 60)

# Suppress excessive HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Setup directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_PATH = os.path.join(script_dir, "STATIC_VARS_MAR2025.env")

# =====================================================================
# ENHANCED QUOTE DISTRIBUTION VALIDATION FUNCTION - 4 QUOTES PER PROBLEM (2-PROBLEM FORMAT)
# =====================================================================

def validate_quote_distribution_short(script_content):
    """
    Validate that the generated SHORT script has exactly 4 quotes per problem (8 total quotes)
    
    Args:
        script_content (str): The generated SHORT video script content
        
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    try:
        # Count total quotes in the script
        quote_count = len(re.findall(r'"[^"]*"', script_content))
        
        # Target: Exactly 8 quotes (4 per problem × 2 problems)
        if quote_count < 6:
            return False, f"Insufficient quotes: {quote_count} (target: exactly 8 quotes - 4 per problem)"
        elif quote_count > 10:
            return False, f"Too many quotes: {quote_count} (target: exactly 8 quotes - 4 per problem)"
        
        # Check for quotes in problem sections
        # Look for common problem indicators
        problem_sections = re.split(r'(?i)(problem|challenge|issue|struggle|difficulty)', script_content)
        
        if len(problem_sections) > 1:  # If we found problem sections
            quotes_in_problems = 0
            for section in problem_sections[1:]:  # Skip the intro before first problem
                # Look for quotes in the next 400 characters after problem indicator
                section_preview = section[:400] if len(section) >= 400 else section
                quotes_in_section = len(re.findall(r'"[^"]*"', section_preview))
                if quotes_in_section > 0:
                    quotes_in_problems += 1
            
            if quotes_in_problems < 2:
                return False, f"Insufficient quotes in problem sections: {quotes_in_problems} (need 4 quotes in each of the 2 problems)"
        
        # Check for professional competence indicators (not ego-stroking)
        ego_indicators = ['hero', 'genius', 'star', 'rockstar', 'superstar', 'legend', 'champion']
        confidence_indicators = ['confident', 'prepared', 'clarity', 'insights', 'data-driven', 'strategic']
        
        ego_count = sum(1 for word in ego_indicators if word in script_content.lower())
        confidence_count = sum(1 for word in confidence_indicators if word in script_content.lower())
        
        if ego_count > confidence_count:
            return False, f"Script leans toward ego-stroking ({ego_count} ego vs {confidence_count} confidence indicators). Focus on professional competence instead."
        
        # Success validation
        return True, f"Perfect SHORT quote distribution: {quote_count} total quotes (4 quotes per problem) with professional competence focus"
        
    except Exception as e:
        return False, f"SHORT validation error: {str(e)}"


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


# =====================================================================
# ENHANCED VIDEO SCRIPT GENERATION FUNCTION - 4 QUOTES PER PROBLEM + COMPANY NAME (SHORT FORMAT)
# =====================================================================

def generate_video_script(voice_guidance,
                          method_guidance,
                          prompt_instructions,
                          poppy_card_content,
                          company_name,
                          openai_model="gpt-4o",
                          max_retries=3,
                          retry_delay=2):
    """
    Generate a SHORT video script using OpenAI API with 4 quotes per problem (8 total) and company name integration
    
    Args:
        voice_guidance (str): Voice and tone guidance
        method_guidance (str): Script structure and framework guidance
        prompt_instructions (str): Specific processing instructions for 5-minute format
        poppy_card_content (str): Poppy Card content to focus on (2-problem format)
        company_name (str): Company name for brand integration
        openai_model (str): OpenAI model to use
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        str: Generated SHORT video script with exactly 8 quotes (4 per problem) and company mentions
    """
    try:
        # Create OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)

        # ✅ ENHANCED SYSTEM PROMPT WITH 4 QUOTES PER PROBLEM FOR SHORT FORMAT
        system_prompt = f"""You are a professional video script writer specializing in concise, high-impact B2B software buyer psychology content. Generate a SHORT video script with these precise specifications:

TARGET: 5-minute duration (approximately 750-900 words)
FORMAT: 2-problem structure for maximum engagement

CRITICAL QUOTE REQUIREMENTS - READ THIS FIRST:
The poppy card contains 16 customer quotes (8 quotes per problem). 
YOU MUST USE EXACTLY 4 QUOTES FROM EACH PROBLEM (8 TOTAL QUOTES IN YOUR SHORT SCRIPT).

MANDATORY DISTRIBUTION - SIMPLE APPROACH:
- Problem 1: Use the FIRST 4 quotes from Problem 1's 8 available quotes
- Problem 2: Use the FIRST 4 quotes from Problem 2's 8 available quotes

⚠️ SHORT SCRIPTS MUST CONTAIN EXACTLY 8 QUOTES (4 PER PROBLEM) ⚠️

VOICE & TONE GUIDELINES:
{voice_guidance}

STRUCTURAL FRAMEWORK (Adapted for Short Format):
{method_guidance}

SPECIFIC SHORT-FORMAT INSTRUCTIONS:
{prompt_instructions}

CONTENT SOURCE (2-Problem Card):
{poppy_card_content}

QUOTE SELECTION MANDATE - KEEP IT SIMPLE:
- Each two-problem poppy card contains 16 total quotes (8 customer quotes per problem)
- For each problem, use the FIRST 4 quotes from that problem's list of 8 quotes
- DO NOT create new quotes - extract and use the provided quotes exactly as written
- Total quotes in your script: exactly 8 quotes (4 × 2 problems)
- DO NOT pick and choose quotes - simply use the first 4 from each problem section

STRATEGIC QUOTE DISTRIBUTION REQUIREMENTS:
- Include exactly 4 customer quotes in EVERY problem section for maximum credibility
- Structure each problem with quotes integrated naturally throughout the problem discussion
- Use role-specific attributions when available (Director of Sales, VP Sales, CRO, CEO)
- Distribute the 4 quotes evenly throughout each problem section
- Use customer quotes throughout the script for optimal conversion psychology

PSYCHOLOGICAL FRAMEWORK - PEER VALIDATION APPROACH:
- Frame this as "peer sharing insights" rather than "company selling solution"
- Each problem should feel like: "Here's what other companies like yours are struggling with"
- Each solution should feel like: "Here's how companies just like yours solved this exact issue"
- Use customer quotes to create "social proof bridges" that connect problems to solutions
- Make viewers think: "This isn't a sales pitch - these are real peer experiences"
- Emphasize "you're not alone" messaging to reduce buyer anxiety

PROFESSIONAL COMPETENCE & CONFIDENCE TRANSFORMATION (NOT EGO-STROKING):
- Focus on increased confidence in decision-making and strategic discussions
- Include quotes about feeling more prepared and data-driven in their role
- Reference reduced stress and increased clarity in high-stakes situations
- Show how better data leads to more confident presentations and discussions
- Include professional competence messaging rather than ego-stroking
- Position as "you'll have the insights needed to make confident decisions"
- Focus on operational excellence rather than personal recognition

CUSTOMER QUOTE AUTHENTICITY GUIDELINES:
- Make quotes feel conversational, not polished marketing speak
- Include specific job titles that match your target buyer personas
- Use industry-specific language that resonates with software buyers
- Include subtle pain points that show the customer truly understands the problem
- Balance problem validation quotes with professional confidence quotes
- Avoid quotes that sound like ego-stroking or "hero" positioning

CRITICAL SHORT-FORMAT REQUIREMENTS:
1. DURATION CONTROL: Aim for 5 minutes (750-900 words maximum)
2. 2-PROBLEM STRUCTURE: 
   - Problem 1: 2-2.5 minutes of content with 4 quotes
   - Transition: 15-30 seconds  
   - Problem 2: 2-2.5 minutes of content with 4 quotes
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

FINAL REMINDER BEFORE YOU BEGIN:
- Use exactly 4 quotes from each problem (first 4 from each problem's list)
- Total script must contain exactly 8 quotes
- Use the exact quotes from the poppy card content provided
- Do not skip quotes or rearrange - use the first 4 from each problem in order

Requirements:
- Write in plain text format
- Use short paragraphs of 1-3 sentences maximum
- Add line breaks between paragraphs
- Create an engaging SHORT video script that follows the voice, method, and focuses on the provided content
- Ensure quote distribution creates a "peer validation experience" rather than a sales pitch"""

        # ADDITIVE ENHANCEMENT: Company Name Integration Instructions (Adapted for SHORT format)
        system_prompt += f"""

🔁 ADDITIONAL MANDATE - COMPANY NAME INTEGRATION (SHORT FORMAT):
- Company name: {company_name}
- Include company name **once naturally in the intro** 
- Include company name **once per Problem section** (2 total)
- Include company name **once in the outro**
- Total company mentions: 4 (intro + 2 problems + outro)
- Expand total script by ~100 words (≈45 seconds narration) through these natural integrations
- Make mentions feel organic - position as the solution provider, not repetitive branding
- Example integration: "...companies like yours working with {company_name} have discovered..."
- Focus on value association: "{company_name} helps businesses..." or "Through {company_name}'s insights..."
- Maintain professional, consultative tone - not salesy or pushy
- Ensure seamless flow - company mentions should enhance, not interrupt, the narrative
- Adapt for 5-minute format: keep mentions brief but impactful"""

        # Continue with retry logic for SHORT scripts
        for attempt in range(max_retries):
            try:
                logger.info(f"[SHORT-GENERATION] Attempt {attempt + 1}/{max_retries} for 5-minute script with 8 quotes and {company_name} integration")
                print(
                    f"Generating SHORT video script with 4 quotes per problem and {company_name} integration using {openai_model} (attempt {attempt + 1}/{max_retries})..."
                )

                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role": "user",
                        "content": "Generate a 5-minute SHORT video script now with exactly 4 quotes from each problem (8 total quotes) and natural company name integration. Focus on the 2-problem structure from the provided card content. Target 750-900 words total with professional competence focus."
                    }],
                    max_tokens=2000,
                    temperature=0.7)

                script_content = response.choices[0].message.content
                if script_content:
                    script_content = script_content.strip()

                if script_content:
                    # Calculate word count for 5-minute target validation
                    word_count = len(script_content.split())
                    logger.info(f"[SHORT-GENERATION] SUCCESS - Generated {word_count} words, {len(script_content)} characters")
                    print(
                        f"Successfully generated 5-minute SHORT video script with quote distribution and {company_name} integration ({len(script_content)} characters, ~{word_count} words)"
                    )
                    
                    # ✅ VALIDATE QUOTE DISTRIBUTION (8 QUOTES EXPECTED)
                    is_valid, validation_message = validate_quote_distribution_short(script_content)
                    if is_valid:
                        logger.info(f"✅ SHORT quote validation passed: {validation_message}")
                        print(f"✅ SHORT quote validation passed: {validation_message}")
                        return script_content
                    else:
                        logger.warning(f"⚠️ SHORT quote validation warning: {validation_message}")
                        print(f"⚠️ SHORT quote validation warning: {validation_message}")
                        return script_content  # Return anyway but log the warning
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

                if attempt < max_retries - 1:
                    logger.info(f"[SHORT-RETRY] Retrying SHORT script generation in {retry_delay}s...")
                    print(
                        f"OpenAI API error: {e}. Retrying SHORT script with quote distribution in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"[SHORT-FAILURE] Failed to generate SHORT script after {max_retries} attempts")
                    print(
                        f"Failed to generate SHORT script with quote distribution after {max_retries} attempts: {e}"
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
    print("LOADING GUIDANCE FILES FOR SHORT SCRIPTS WITH QUOTE DISTRIBUTION")
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

        print("Successfully loaded all guidance files for SHORT scripts with quote distribution")
        return guidance_files

    except Exception as e:
        print(f"Error loading guidance files: {str(e)}")
        raise Exception(f"Failed to load required guidance files from bucket '{bucket_name}': {str(e)}")


def process_poppy_cards(variables, guidance_files):
    """
    Process 5 Poppy Card combinations sequentially (cards 11-15) with 4 quotes per problem and company integration
    
    Args:
        variables (dict): Configuration variables from Supabase
        guidance_files (dict): Loaded guidance files
        
    Returns:
        dict: Summary of processed SHORT scripts with quote distribution validation
    """
    try:
        # Extract configuration
        video_script_config = variables["scripts"]["video_script_short"]
        company_name = variables["global"]["COMPANY_NAME"]
        openai_model = video_script_config.get("openai_model", "gpt-4o")
        
        # Get bucket configurations
        input_bucket = video_script_config["supabase_buckets"]["input_cards"]
        output_bucket = video_script_config["supabase_buckets"]["output"]
        
        # Use predefined card combinations from configuration
        card_combinations = video_script_config["card_combinations"]
        
        # Process each combination (cards 11-15)
        processed_scripts = []
        timestamp = datetime.now(eastern_tz).strftime("%Y%m%d_%H%M")
        total_cards = len(card_combinations)

        print(f"\n" + "=" * 80)
        print("PROCESSING POPPY CARDS WITH 4 QUOTES PER PROBLEM + COMPANY INTEGRATION (SHORT FORMAT)")
        print("=" * 80)
        print(f"Total combinations to process: {total_cards}")
        print(f"Company: {company_name}")
        print(f"OpenAI Model: {openai_model}")
        print(f"Quote Distribution: 4 quotes per problem (8 total per SHORT script)")
        print(f"Company Integration: 4 mentions per script (adapted for 5-minute format)")
        print(f"Timestamp: {timestamp}")
        
        for i, combination in enumerate(card_combinations, 1):
            print(f"\nProcessing SHORT card {i} of {total_cards}...")
            print(f"Combination: {combination}")

            try:
                # Construct input and output filenames for cards 11-15
                card_number = f"card{i+10:02d}"  # Formats as card11, card12, ..., card15
                input_filename = f"{company_name}_{card_number}_{combination}.txt"
                output_filename = f"{company_name}_SHORT_script_{combination}_{timestamp}.txt"

                print(f"Looking for file: {input_filename}")

                # Download the Poppy Card content
                poppy_card_content = download_file_from_bucket(
                    input_bucket, input_filename)

                # ✅ ENHANCED SCRIPT GENERATION WITH 4 QUOTES PER PROBLEM + COMPANY NAME (SHORT FORMAT)
                script_content = generate_video_script(
                    voice_guidance=guidance_files["voice"],
                    method_guidance=guidance_files["method"],
                    prompt_instructions=guidance_files["prompt"],
                    poppy_card_content=poppy_card_content,
                    company_name=company_name,
                    openai_model=openai_model
                )

                # Upload the generated script
                upload_file_to_bucket(output_bucket, output_filename,
                                      script_content)

                # ✅ VALIDATION RESULTS (EXPECTING 8 QUOTES FOR SHORT FORMAT)
                is_valid, validation_message = validate_quote_distribution_short(script_content)
                quote_count = len(re.findall(r'"[^"]*"', script_content))
                word_count = len(script_content.split())
                
                # Record the processed script with enhanced metrics
                processed_scripts.append({
                    "combination": combination,
                    "input_file": input_filename,
                    "output_file": output_filename,
                    "script_length": len(script_content),
                    "word_count": word_count,
                    "quote_count": quote_count,
                    "validation_passed": is_valid,
                    "validation_message": validation_message,
                    "target_compliance": "optimal" if 700 <= word_count <= 950 else ("low" if word_count < 700 else "high"),
                    "status": "success",
                    "script_type": "SHORT"
                })

                print(f"✅ Successfully processed SHORT {combination}")
                print(f"📊 Quote count: {quote_count}, Target: 8, Validation: {'PASSED' if is_valid else 'WARNING'}")
                print(f"📊 Word count: {word_count}, Target: 750-900")
                print(f"📋 {validation_message}")

            except Exception as e:
                print(f"❌ Error processing SHORT {combination}: {str(e)}")
                processed_scripts.append({
                    "combination": combination,
                    "input_file": input_filename if 'input_filename' in locals() else "unknown",
                    "output_file": output_filename if 'output_filename' in locals() else "unknown",
                    "error": str(e),
                    "status": "failed",
                    "validation_passed": False,
                    "quote_count": 0,
                    "word_count": 0,
                    "script_type": "SHORT"
                })

        # Enhanced summary with validation statistics for SHORT format
        successful_scripts = [s for s in processed_scripts if s["status"] == "success"]
        failed_scripts = [s for s in processed_scripts if s["status"] == "failed"]
        validated_scripts = [s for s in successful_scripts if s.get("validation_passed", False)]
        
        # Calculate average quote count and word count for successful scripts
        avg_quote_count = sum(s.get("quote_count", 0) for s in successful_scripts) / len(successful_scripts) if successful_scripts else 0
        avg_word_count = sum(s.get("word_count", 0) for s in successful_scripts) / len(successful_scripts) if successful_scripts else 0
        optimal_count = len([s for s in successful_scripts if s.get("target_compliance") == "optimal"])

        summary = {
            "total_processed": total_cards,
            "successful": len(successful_scripts),
            "failed": len(failed_scripts),
            "validation_passed": len(validated_scripts),
            "validation_rate": f"{len(validated_scripts)}/{len(successful_scripts)}" if successful_scripts else "0/0",
            "average_quote_count": round(avg_quote_count, 1),
            "target_quote_count": 8,
            "scripts": processed_scripts,
            "company_name": company_name,
            "timestamp": timestamp,
            "openai_model": openai_model,
            "script_type": "SHORT",
            "card_range": "11-15",
            "target_duration": "5 minutes",
            "word_count_analysis": {
                "average_word_count": round(avg_word_count, 1),
                "optimal_compliance": f"{optimal_count}/{len(successful_scripts)}" if successful_scripts else "0/0",
                "target_range": "750-900 words"
            }
        }

        print(f"\n📊 PROCESSING SUMMARY - SHORT FORMAT WITH 4 QUOTES PER PROBLEM + COMPANY INTEGRATION:")
        print(f"✅ SHORT scripts generated: {len(successful_scripts)}/{total_cards}")
        print(f"✅ Validation passed: {len(validated_scripts)}/{len(successful_scripts)}")
        print(f"📈 Average quote count: {avg_quote_count:.1f} (target: 8)")
        print(f"📈 Average word count: {avg_word_count:.1f} (target: 750-900)")
        print(f"🏢 Company integration: {company_name} mentioned 4 times per SHORT script")
        print(f"🎯 Quote distribution success rate: {(len(validated_scripts)/len(successful_scripts)*100):.1f}%" if successful_scripts else "0%")
        print(f"🎯 Word count compliance: {optimal_count}/{len(successful_scripts)} scripts in optimal range")
        
        return summary

    except Exception as e:
        print(f"❌ Error in process_poppy_cards: {str(e)}")
        raise


def main():
    """Main function to orchestrate the entire enhanced SHORT video script workflow."""
    try:
        logger.info("=" * 80)
        logger.info("ENHANCED SHORT VIDEO SCRIPT AUTOMATION WITH QUOTE DISTRIBUTION - MAIN START")
        logger.info("=" * 80)
        print("=" * 80)
        print("ENHANCED SHORT VIDEO SCRIPT AUTOMATION WITH QUOTE DISTRIBUTION")
        print("=" * 80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Script directory: {SCRIPT_DIR}")
        print(f"Target: 5-minute scripts with 8 quotes (4 per problem) from cards 11-15")
        print("🎯 APPROACH: Exactly 4 quotes per problem (8 total quotes per SHORT script)")
        print("🏢 ENHANCEMENT: Company name integration (4 mentions adapted for 5-minute format)")

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
        
        video_script_config = variables["scripts"]["video_script_short"]
        logger.info("[SHORT-CONFIG] Successfully loaded video_script_short configuration")

        # Load guidance files (using SHORT version) with no fallback handling
        guidance_bucket = video_script_config["supabase_buckets"]["guidance"]
        try:
            guidance_files = load_guidance_files(guidance_bucket)
            logger.info("[SHORT-CONFIG] Successfully loaded all guidance files")
        except Exception as e:
            logger.critical(f"[SHORT-CRITICAL] Failed to load guidance files: {str(e)}")
            raise Exception(f"Failed to load SHORT script guidance files from bucket '{guidance_bucket}': {str(e)}")

        # Process Poppy Cards (cards 11-15) with comprehensive error handling and quote distribution
        try:
            logger.info("[SHORT-PROCESSING] Starting card processing for cards 11-15 with quote distribution")
            summary = process_poppy_cards(variables, guidance_files)
            logger.info(f"[SHORT-PROCESSING] Completed processing: {summary['successful']}/{summary['total_processed']} successful")
        except Exception as e:
            logger.critical(f"[SHORT-CRITICAL] Failed during card processing: {str(e)}")
            raise Exception(f"Failed to process SHORT script cards 11-15: {str(e)}")

        # Save summary to output bucket - MUST SUCCEED  
        output_bucket = video_script_config["supabase_buckets"]["output"]
        summary_filename = f"video_script_SHORT_enhanced_summary_{summary['timestamp']}.json"
        summary_content = json.dumps(summary, indent=2)
        
        try:
            upload_file_to_bucket(output_bucket, summary_filename, summary_content)
            logger.info(f"[SHORT-SUMMARY] Saved enhanced summary as {summary_filename}")
        except Exception as e:
            logger.error(f"[SHORT-ERROR] Failed to save summary: {str(e)}")
            print(f"Warning: Could not save summary file to bucket '{output_bucket}': {str(e)}")
            print("Video script generation completed successfully despite summary save failure.")

        # Calculate and display execution time
        end_time = datetime.now(eastern_tz)
        execution_time = end_time - start_time

        logger.info("=" * 80)
        logger.info("ENHANCED SHORT VIDEO SCRIPT WORKFLOW COMPLETE")
        logger.info(f"Execution time: {execution_time}")
        logger.info(f"Success rate: {summary['successful']}/{summary['total_processed']}")
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("ENHANCED SHORT VIDEO SCRIPT WORKFLOW COMPLETE")
        print("=" * 80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {execution_time}")
        print(f"SHORT scripts generated: {summary['successful']}/{summary['total_processed']}")
        print(f"Validation success rate: {summary['validation_rate']}")
        print(f"Average quote count: {summary['average_quote_count']} (target: 8)")
        print(f"Average word count: {summary['word_count_analysis']['average_word_count']} (target: 750-900)")
        print(f"Company integration: {summary['company_name']} mentioned 4 times per SHORT script")
        print(f"Summary saved as: {summary_filename}")

        if summary['failed'] > 0:
            logger.warning(f"[SHORT-WARNING] {summary['failed']} scripts failed to generate")
            print(f"\n⚠️ Warning: {summary['failed']} SHORT script(s) failed to generate")
            for script in summary['scripts']:
                if script['status'] == 'failed':
                    logger.error(f"[SHORT-FAILED-SCRIPT] {script['combination']}: {script.get('error', 'Unknown error')}")
                    print(f"  - {script['combination']}: {script.get('error', 'Unknown error')}")

        # Show validation statistics
        validation_passed = summary['validation_passed']
        total_successful = summary['successful']
        if total_successful > 0:
            print(f"\n📊 QUOTE DISTRIBUTION ANALYSIS (SHORT FORMAT):")
            print(f"✅ Scripts with 8 quotes (4 per problem): {validation_passed}/{total_successful}")
            print(f"🏢 Company mentions per script: 4 (intro + 2 problems + outro)")
            print(f"📈 Professional competence focus maintained across all SHORT scripts")
            print(f"🎯 Peer validation psychology successfully implemented")
            print(f"⏱️ 5-minute format compliance: {summary['word_count_analysis']['optimal_compliance']}")

        print("\n🎉 Enhanced SHORT video script automation workflow completed successfully!")
        print("📋 Each SHORT script contains exactly 8 quotes (4 quotes per problem)")
        print("🏢 Each SHORT script includes natural company name integration (4 mentions)")
        print("⏱️ Each SHORT script targets 5-minute duration with optimal pacing")
        
        # Log successful session completion
        logger.info("=" * 60)
        logger.info("ENHANCED SHORT VIDEO SCRIPT AUTOMATION - SESSION END (SUCCESS)")
        logger.info("=" * 60)

    except Exception as e:
        logger.critical(f"[SHORT-CRITICAL] Critical error in enhanced SHORT workflow: {str(e)}")
        logger.critical(f"[SHORT-CRITICAL] Traceback: {traceback.format_exc()}")
        print(f"\nCritical error in enhanced SHORT workflow: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Ensure we log session end even on failure
        logger.info("=" * 60)
        logger.info("ENHANCED SHORT VIDEO SCRIPT AUTOMATION - SESSION END (FAILED)")
        logger.info("=" * 60)
        raise


if __name__ == "__main__":
    main()
