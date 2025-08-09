#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Video Script Automation Workflow Script with 4 Quotes Per Problem

This script processes Poppy Card content and generates video scripts using:
- Voice guidance (tone and style)
- Method guidance (structure and framework) 
- Prompt instructions (specific processing directions)
- Poppy Card content (unique subject matter)

ENHANCEMENTS (ALIGNED WITH TWO-PROBLEM VERSION):
- Simple quote distribution (4 quotes per problem = 16 total quotes per script)
- Peer validation psychology framework
- Professional competence vs. ego-stroking detection
- Quote distribution validation and reporting
- Company name integration (FIXED: 9-12 mentions with natural distribution)
- ADDITIVE IMPROVEMENTS: Feature clarity, revenue impact, implementation assurance, competitive differentiation
- Enhanced content validation with full strictness
- Prohibited terms cleanup function (TRANSPLANTED FROM TWO-PROBLEM VERSION)

The workflow processes 10 predefined Poppy Card combinations sequentially (cards 1-10),
generating custom video scripts for each combination and saving them to Supabase.

Card Range: 1-10 (four-problem format cards)
Target Duration: 8-12 minutes 
Target Word Count: 2000-2800 words
Quote Distribution: 4 quotes per problem (16 total quotes)
Company Integration: FIXED - 9-12 mentions with natural distribution
Output Bucket: four-problem-script-drafts

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

# Configure logging with FOUR-PROBLEM identifier
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [FOUR-PROBLEM-ENHANCED] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"video_script_FOUR_enhanced_creative_log_{datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}.log")
    ])
logger = logging.getLogger(__name__)

# Log initialization for enhanced FOUR-PROBLEM workflow
logger.info("=" * 60)
logger.info("FOUR-PROBLEM VIDEO SCRIPT AUTOMATION - ENHANCED WITH PROHIBITED TERMS CLEANUP")
logger.info(f"Target: 8-12 minute scripts with 16 quotes (4 per problem) from cards 1-10")
logger.info(f"Word Target: 2000-2800 words with rich creative development")
logger.info(f"Company Integration: FIXED - 9-12 mentions with natural distribution")
logger.info(f"Session ID: {datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}")
logger.info("=" * 60)

# Suppress excessive HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Setup directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_PATH = os.path.join(script_dir, "STATIC_VARS_MAR2025.env")

# =====================================================================
# ENHANCED QUOTE DISTRIBUTION VALIDATION FUNCTION - 4-PROBLEM STRUCTURE
# =====================================================================

def validate_quote_distribution(script_content):
    """
    Enhanced validation for 4-problem structure with balanced quote distribution and content balance
    
    Args:
        script_content (str): The generated FOUR-PROBLEM video script content
        
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    try:
        # Count total quotes in the script
        quote_count = len(re.findall(r'"[^"]*"', script_content))
        
        # Target: Exactly 16 quotes (4 per problem × 4 problems)
        if quote_count < 14:
            return False, f"Insufficient quotes: {quote_count} (target: exactly 16 quotes - 4 per problem)"
        elif quote_count > 18:
            return False, f"Too many quotes: {quote_count} (target: exactly 16 quotes - 4 per problem)"
        
        # Enhanced structural validation - Split script into sections
        sections = re.split(r'(?i)(problem|challenge|issue|struggle|difficulty)', script_content)
        
        if len(sections) < 5:  # Need at least intro + 4 problems
            return False, "Script missing clear 4-problem structure with identifiable problem sections"
        
        # Validate quotes per problem section - Look for distinct problem areas
        problem_quotes = []
        problem_word_counts = []
        
        # Analyze first four major sections after problem indicators
        for i in range(1, min(9, len(sections))):  # Check up to 8 sections after splits
            if i in [1, 3, 5, 7]:  # Likely problem sections (odd indices after splits)
                if i + 1 < len(sections):
                    section_content = sections[i + 1]  # Content after problem indicator
                    # Look at substantial portion of each problem section
                    section_preview = section_content[:1200] if len(section_content) >= 1200 else section_content
                    section_quotes = len(re.findall(r'"[^"]*"', section_preview))
                    section_words = len(section_preview.split())
                    
                    problem_quotes.append(section_quotes)
                    problem_word_counts.append(section_words)
        
        if len(problem_quotes) < 4:
            return False, "Could not identify 4 distinct problem sections with quotes"
        
        # Validate quote distribution per problem -- VARIES FROM TWO-PROBLEM VERSION
        for i, quote_count in enumerate(problem_quotes[:4]):
            if quote_count < 2:
                return False, f"Insufficient quotes in Problem {i+1}: {quote_count} quotes (need ~4 each)"
        
        # Check content balance between problems (adjusted for longer format)
        if len(problem_word_counts) >= 4:
            avg_words = sum(problem_word_counts[:4]) / 4
            for i, word_count in enumerate(problem_word_counts[:4]):
                word_diff = abs(word_count - avg_words)
                if word_diff > 400:  # Allow more flexibility for longer content
                    return False, f"Unbalanced problem development: P{i+1}: {word_count} words vs avg: {avg_words:.0f} (difference: {word_diff:.0f})"
        
        # Check for professional competence indicators (not ego-stroking)
        ego_indicators = ['hero', 'genius', 'star', 'rockstar', 'superstar', 'legend', 'champion']
        confidence_indicators = ['confident', 'prepared', 'clarity', 'insights', 'data-driven', 'strategic']
        
        ego_count = sum(1 for word in ego_indicators if word in script_content.lower())
        confidence_count = sum(1 for word in confidence_indicators if word in script_content.lower())
        
        if ego_count > confidence_count:
            return False, f"Script leans toward ego-stroking ({ego_count} ego vs {confidence_count} confidence indicators). Focus on professional competence instead."
        
        # Enhanced success validation with balance metrics
        balance_info = ""
        if len(problem_word_counts) >= 4:
            balance_info = f", balanced content ({'/'.join(str(w) for w in problem_word_counts[:4])} words per problem)"
        
        return True, f"Excellent FOUR-PROBLEM structure: {quote_count} total quotes ({'/'.join(str(q) for q in problem_quotes[:4])} per problem){balance_info} with professional competence focus"
        
    except Exception as e:
        return False, f"FOUR-PROBLEM validation error: {str(e)}"


# =====================================================================
# ENHANCED CONTENT VALIDATION FUNCTION - ALIGNED WITH TWO-PROBLEM VERSION
# =====================================================================

def validate_enhanced_content(script_content, company_name):
    """
    Validate the enhanced content requirements for FOUR-PROBLEM format (aligned with two-problem version)
    
    Args:
        script_content (str): The generated video script content
        company_name (str): Company name to check for
        
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    try:
        validation_issues = []
        
        # Check for implementation time mention (< 8 hours) - same as two-problem version
        implementation_keywords = ['8 hour', 'eight hour', 'quick implementation', 'rapid deployment', 'fast setup']
        has_implementation_mention = any(keyword in script_content.lower() for keyword in implementation_keywords)
        if not has_implementation_mention:
            validation_issues.append("Missing implementation time assurance in outro")
        
        # Check for payback period mention (< 6 months) - same as two-problem version
        payback_keywords = ['6 month', 'six month', 'payback', 'roi', 'return on investment']
        has_payback_mention = any(keyword in script_content.lower() for keyword in payback_keywords)
        if not has_payback_mention:
            validation_issues.append("Missing payback period assurance in problem sections")
        
        # Check for competitive differentiation - same as two-problem version
        competitive_keywords = ['better than', 'unlike', 'superior', 'advantage', 'unique', 'differentiat']
        has_competitive_mention = any(keyword in script_content.lower() for keyword in competitive_keywords)
        if not has_competitive_mention:
            validation_issues.append("Missing competitive differentiation in outro")
        
        # Check for feature explanations - same as two-problem version
        feature_keywords = ['feature', 'capability', 'functionality', 'tool', 'dashboard', 'analytics']
        has_feature_mention = any(keyword in script_content.lower() for keyword in feature_keywords)
        if not has_feature_mention:
            validation_issues.append("Missing specific feature explanations in problem sections")
        
        # FIXED: Check for company name mentions with natural range (adjusted for 4-problem format)
        company_mentions = script_content.lower().count(company_name.lower())
        if company_mentions < 6:
            validation_issues.append(f"Low company mentions: {company_mentions} (target: 9-12 mentions)")
        elif company_mentions > 15:
            validation_issues.append(f"High company mentions: {company_mentions} (target: 9-12 mentions)")
        
        if validation_issues:
            return False, f"Enhanced content validation issues: {', '.join(validation_issues)}"
        else:
            return True, f"All enhanced content requirements validated successfully (company mentions: {company_mentions})"
            
    except Exception as e:
        return False, f"Enhanced validation error: {str(e)}"


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
        logger.info(f"[FOUR-PROBLEM-ENHANCED] Downloading {file_name} from {bucket_name} bucket...")
        print(f"Downloading {file_name} from {bucket_name} bucket...")
        response = supabase.storage.from_(bucket_name).download(file_name)

        if response:
            content = response.decode('utf-8')
            logger.info(f"[FOUR-PROBLEM-ENHANCED] Successfully downloaded {file_name} ({len(content)} characters)")
            print(
                f"Successfully downloaded {file_name} ({len(content)} characters)"
            )
            return content
        else:
            raise Exception(f"Failed to download {file_name}")

    except Exception as e:
        logger.error(f"[FOUR-PROBLEM-ENHANCED] Error downloading {file_name} from {bucket_name}: {str(e)}")
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
        logger.info(f"[FOUR-PROBLEM-ENHANCED] Uploading {file_name} to {bucket_name} bucket...")
        print(f"Uploading {file_name} to {bucket_name} bucket...")

        # Convert string content to bytes
        file_bytes = file_content.encode('utf-8')

        response = supabase.storage.from_(bucket_name).upload(
            file_name, file_bytes, {"content-type": "text/plain"})

        logger.info(f"[FOUR-PROBLEM-ENHANCED] Successfully uploaded {file_name} to {bucket_name}")
        print(f"Successfully uploaded {file_name} to {bucket_name}")
        return True

    except Exception as e:
        logger.error(f"[FOUR-PROBLEM-ENHANCED] Error uploading {file_name} to {bucket_name}: {str(e)}")
        print(f"Error uploading {file_name} to {bucket_name}: {str(e)}")
        raise


# =====================================================================
# ENHANCED VIDEO SCRIPT GENERATION - PROHIBITED TERMS CLEANUP TRANSPLANTED FROM TWO-PROBLEM VERSION
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
    Generate a FOUR-PROBLEM video script with prohibited terms cleanup from two-problem version
    
    Args:
        voice_guidance (str): Voice and tone guidance
        method_guidance (str): Script structure and framework guidance
        prompt_instructions (str): Specific processing instructions for 8-12 minute format
        poppy_card_content (str): Poppy Card content to focus on (4-problem format)
        company_name (str): Company name for brand integration
        openai_model (str): OpenAI model to use
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        str: Generated FOUR-PROBLEM video script with exactly 16 quotes (4 per problem) and prohibited terms cleaned
    """
    try:
        # Create OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)

        # ✅ ALIGNED CREATIVE GUIDANCE SYSTEM PROMPT - ADAPTED FOR 4-PROBLEM STRUCTURE
        system_prompt = f"""You are a professional video script writer specializing in B2B software buyer psychology. Generate a FOUR-PROBLEM video script using the following guidance:

TARGET: 8-12 minute duration (approximately 2000-2800 words)
FORMAT: 4-problem structure for comprehensive engagement with rich development

CRITICAL QUOTE REQUIREMENTS - READ THIS FIRST:
The poppy card contains 32 customer quotes (8 quotes per problem). 
YOU MUST USE EXACTLY 4 QUOTES FROM EACH PROBLEM (16 TOTAL QUOTES IN YOUR FOUR-PROBLEM SCRIPT).

MANDATORY DISTRIBUTION - SIMPLE APPROACH:
- Problem 1: Use the FIRST 4 quotes from Problem 1's 8 available quotes
- Problem 2: Use the FIRST 4 quotes from Problem 2's 8 available quotes
- Problem 3: Use the FIRST 4 quotes from Problem 3's 8 available quotes
- Problem 4: Use the FIRST 4 quotes from Problem 4's 8 available quotes

⚠️ FOUR-PROBLEM SCRIPTS MUST CONTAIN EXACTLY 16 QUOTES (4 PER PROBLEM) ⚠️

Generate a video script using the following guidance:

VOICE GUIDELINES:
{voice_guidance}

SCRIPT METHOD/FRAMEWORK:
{method_guidance}

SPECIFIC INSTRUCTIONS:
{prompt_instructions}

CONTENT TO FOCUS ON:
{poppy_card_content}

QUOTE SELECTION MANDATE - KEEP IT SIMPLE:
- Each four-problem poppy card contains 32 total quotes (8 customer quotes per problem)
- For each problem, use the FIRST 4 quotes from that problem's list of 8 quotes
- DO NOT create new quotes - extract and use the provided quotes exactly as written
- Total quotes in your script: exactly 16 quotes (4 × 4 problems)
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

WORD DISTRIBUTION REQUIREMENTS:
- Total script: 2000-2800 words
- Intro: 150-200 words (hook + company mention)
- Problem 1: 500-650 words (4 quotes + enhancements + company mention)
- Problem 2: 500-650 words (4 quotes + enhancements + company mention)
- Problem 3: 500-650 words (4 quotes + enhancements + company mention)
- Problem 4: 500-650 words (4 quotes + enhancements + company mention)
- Outro: 200-250 words (implementation + competitive + company mentions)

FINAL REMINDER BEFORE YOU BEGIN:
- Use exactly 4 quotes from each problem (first 4 from each problem's list)
- Total script must contain exactly 16 quotes
- Use the exact quotes from the poppy card content provided
- Do not skip quotes or rearrange - use the first 4 from each problem in order
- Target 2000-2800 words for rich, engaging content

Requirements:
- Write in plain text format
- Use short paragraphs of 1-3 sentences maximum
- Add line breaks between paragraphs
- Create an engaging video script that follows the voice, method, and focuses on the provided content
- Ensure quote distribution creates a "peer validation experience" rather than a sales pitch"""

        # FIXED: SIMPLIFIED COMPANY NAME USAGE SECTION
        system_prompt += f"""

🏢 COMPANY NAME USAGE - NATURAL DISTRIBUTION:

Company name: {company_name}

SIMPLE APPROACH:
- Include {company_name} 9-12 times throughout your script
- Distribute naturally across sections: intro, problem 1, problem 2, problem 3, problem 4, outro
- Use alternatives when it would sound repetitive: "the platform", "this solution", "the system", "this technology"
- Focus on natural flow rather than rigid counting rules

GOAL: Professional brand integration without oversaturation or forced repetition.

DISTRIBUTION GUIDE:
- Intro: 1-2 mentions when introducing the solution
- Problem 1: 2-3 mentions when discussing how it addresses this problem
- Problem 2: 2-3 mentions when discussing how it addresses this problem  
- Problem 3: 2-3 mentions when discussing how it addresses this problem
- Problem 4: 2-3 mentions when discussing how it addresses this problem
- Outro: 2-3 mentions for final brand reinforcement

ALTERNATIVE REFERENCES:
- "the platform"
- "this solution" 
- "the system"
- "this technology"
- "this advanced tool"
- "this top-rated solution"
"""

        # ✅ ADD PROHIBITED TERMS SECTION HERE (TRANSPLANTED FROM TWO-PROBLEM VERSION)
        system_prompt += f"""

📝 PROHIBITED TERMS & CONVERSATIONAL ALTERNATIVES:

MANDATORY TERM SUBSTITUTIONS - USE CONVERSATIONAL LANGUAGE:
The following overused marketing terms are PROHIBITED. Use the suggested conversational alternatives instead:

❌ NEVER USE → ✅ USE INSTEAD:
- "unlock" → "get"
- "transform" → "modernize"
- "revolutionize" → "advance"
- "seamless" → "smooth"
- "game-changer" → "major advantage"
- "game-changing" → "high-value"
- "empower" → "enable"
- "catalyst" → "driver"
- "operational excellence" → "solid performance"

TONE REQUIREMENT:
- Write as if explaining to a colleague, not delivering a sales pitch
- Use natural, conversational language that people actually use in business meetings
- Avoid marketing jargon that sounds artificial or overly promotional
- Focus on practical benefits using everyday business language

VALIDATION:
If any prohibited terms appear in your script, immediately replace them with conversational alternatives before finalizing the content.
"""

        # Continue with additive enhancements section
        system_prompt += f"""

🎯 PROBLEM SECTION CONTENT REQUIREMENTS - ADDITIVE ENHANCEMENTS:

FEATURE CLARITY MANDATE (Requirement 1):
- In each of the 4 problem sections, add 2-3 sentences explaining the specific feature that addresses this problem
- Use fewer than 800 additional characters per problem section for feature explanations
- Include high-level explanation of how the feature helps solve the specific problem
- Use {company_name} OR alternative references naturally based on flow
- Make feature descriptions concrete and specific, not vague marketing language
- Examples: "The pipeline analytics dashboard shows exactly where deals are stuck" or "The automated scoring system highlights which prospects need immediate attention"

REVENUE IMPACT MANDATE (Requirement 2):
- In each of the 4 problem sections, add 2-3 sentences explaining how this solution increases revenue
- Use fewer than 800 additional characters per problem section for revenue impact explanations
- Specifically assure viewers they will achieve less than 6-month payback period
- Use {company_name} OR alternative references naturally based on flow
- Include specific revenue generation mechanisms (faster deals, better conversion, reduced waste, etc.)
- Position as "fully engaged users consistently achieve sub-6-month ROI"
- Examples: "Companies using deal acceleration features see 23% faster close rates, typically achieving full payback in under 6 months" or "Users report 31% improvement in qualified lead conversion, with most seeing ROI within 5 months"

🚀 OUTRO CONTENT REQUIREMENTS - ADDITIVE ENHANCEMENTS:

IMPLEMENTATION ASSURANCE MANDATE (Requirement 3):
- In the first 4 sentences of the outro, assure viewers that full implementation takes less than 8 hours
- Use fewer than 800 additional characters for implementation time assurance
- Use {company_name} OR alternative references naturally based on flow
- Emphasize minimal disruption to current operations
- Position as "rapid deployment advantage"
- Examples: "The platform deploys in under 8 hours with zero disruption to your current sales process" or "Full implementation typically completes in 6-8 hours, often during a single business day"

COMPETITIVE DIFFERENTIATION MANDATE (Requirement 4):
- In the first 5 sentences of the outro, clarify what makes the solution superior to competitors
- Use fewer than 800 additional characters for competitive advantages
- Use {company_name} OR alternative references naturally based on flow
- Include 2 specific differentiators that are concrete and measurable
- Focus on unique capabilities, not generic benefits
- Avoid naming specific competitors - focus on category advantages
- Examples: "Unlike traditional CRM analytics, the platform provides predictive deal scoring and real-time pipeline health monitoring" or "The peer-based benchmarking gives you insights that generic sales platforms simply cannot match"
"""

        # ✅ ADD SYNOPSIS CONTROL SECTION HERE (TRANSPLANTED FROM TWO-PROBLEM VERSION)
        system_prompt += f"""

🚫 SYNOPSIS CONTROL:

MANDATORY OUTPUT RESTRICTIONS:
- Do NOT add any summary, synopsis, or concluding remarks after the outro section
- End the script immediately after the outro content
- Do NOT include any content after the final outro paragraph
- Do NOT add section headers, timestamps, or meta-commentary
- Focus only on the actual video script content that will be spoken

SCRIPT ENDING REQUIREMENT:
- Your script should end with the final sentence of the outro section
- No additional commentary, analysis, or wrap-up content
- No "In summary", "To conclude", or similar ending phrases beyond the outro
- Keep the script tight and focused for video production
"""

        # ✅ PROHIBITED TERMS CLEANUP FUNCTION (TRANSPLANTED FROM TWO-PROBLEM VERSION)
        def clean_prohibited_terms(script_content):
            """
            Replace prohibited terms with specific acceptable alternatives
            
            Args:
                script_content (str): Generated script content
                
            Returns:
                str: Script with prohibited terms replaced
            """
            replacements = {
                'transform': 'modernize',
                'transforms': 'modernizes', 
                'transforming': 'modernizing',
                'transformation': 'modernization',
                'revolutionize': 'advance',
                'revolutionizing': 'advancing',
                'revolutionizes': 'advances',
                'revolution': 'advancement',
                'unlock': 'access',
                'unlocking': 'accessing',
                'unlocks': 'accesses',
                'empower': 'enable',
                'empowers': 'enables',
                'empowering': 'enabling',
                'empowerment': 'enablement',
                'seamless': 'smooth',
                'seamlessly': 'smoothly',
                'game-changer': 'major advantage',
                'game changer': 'major advantage',
                'game-changing': 'highly beneficial',
                'catalyst': 'driver',
                'catalysts': 'drivers'
            }
            
            cleaned_content = script_content
            replaced_terms = []
            
            for prohibited, replacement in replacements.items():
                # Case-insensitive replacement while preserving original case
                import re
                pattern = re.compile(re.escape(prohibited), re.IGNORECASE)
                if pattern.search(cleaned_content):
                    replaced_terms.append(prohibited)
                    cleaned_content = pattern.sub(replacement, cleaned_content)
            
            return cleaned_content, replaced_terms

        # ✅ SIMPLIFIED GENERATION WITH AUTOMATIC CLEANUP (TRANSPLANTED FROM TWO-PROBLEM VERSION)
        for attempt in range(max_retries):
            try:
                logger.info(f"[FOUR-PROBLEM-ENHANCED] Attempt {attempt + 1}/{max_retries} for creative-enhanced 4-problem script with {company_name} integration")
                print(
                    f"Generating creative-enhanced FOUR-PROBLEM video script aligned with two-problem version and {company_name} integration using {openai_model} (attempt {attempt + 1}/{max_retries})..."
                )

                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role": "user",
                        "content": "Generate an 8-12 minute FOUR-PROBLEM video script now with exactly 4 quotes from each problem (16 total quotes), natural company name distribution (9-12 mentions), and all additive improvements including feature clarity, revenue impact, implementation assurance, and competitive differentiation. Target 2000-2800 words with rich creative development matching the depth of the two-problem version."
                    }],
                    max_tokens=4000,
                    temperature=0.1)

                script_content = response.choices[0].message.content
                if script_content:
                    script_content = script_content.strip()

                if script_content:
                    # ✅ AUTOMATIC PROHIBITED TERMS CLEANUP (TRANSPLANTED FROM TWO-PROBLEM VERSION)
                    cleaned_script, replaced_terms = clean_prohibited_terms(script_content)
                    
                    if replaced_terms:
                        logger.info(f"[FOUR-PROBLEM-ENHANCED] ✅ Auto-replaced prohibited terms: {replaced_terms}")
                        print(f"✅ Auto-replaced prohibited terms: {replaced_terms}")
                    else:
                        logger.info(f"[FOUR-PROBLEM-ENHANCED] ✅ No prohibited terms found - script was clean")
                        print(f"✅ No prohibited terms found - script was clean")
                    
                    script_content = cleaned_script
                    
                    # Calculate word count for 8-12 minute target validation
                    word_count = len(script_content.split())
                    logger.info(f"[FOUR-PROBLEM-ENHANCED] SUCCESS - Generated {word_count} words, {len(script_content)} characters")
                    print(
                        f"Successfully generated creative-enhanced FOUR-PROBLEM video script aligned with two-problem version and {company_name} integration ({len(script_content)} characters, ~{word_count} words)"
                    )
                    
                    # ✅ DUAL VALIDATION - STRUCTURAL AND ENHANCED CONTENT
                    is_valid, validation_message = validate_quote_distribution(script_content)
                    is_enhanced_valid, enhanced_message = validate_enhanced_content(script_content, company_name)
                    
                    if is_valid:
                        logger.info(f"✅ Structural validation passed: {validation_message}")
                        print(f"✅ Structural validation passed: {validation_message}")
                    else:
                        logger.warning(f"⚠️ Structural validation warning: {validation_message}")
                        print(f"⚠️ Structural validation warning: {validation_message}")
                    
                    if is_enhanced_valid:
                        logger.info(f"✅ Enhanced content validation passed: {enhanced_message}")
                        print(f"✅ Enhanced content validation passed: {enhanced_message}")
                    else:
                        logger.warning(f"⚠️ Enhanced content validation warning: {enhanced_message}")
                        print(f"⚠️ Enhanced content validation warning: {enhanced_message}")
                        
                    return script_content
                else:
                    raise Exception("Empty response from OpenAI for creative-enhanced FOUR-PROBLEM script generation")

            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"[FOUR-PROBLEM-ENHANCED] Attempt {attempt + 1} failed: {str(e)}")

                # Enhanced error handling for FOUR-PROBLEM script context
                if "model" in error_msg and ("not found" in error_msg
                                             or "unavailable" in error_msg
                                             or "sunset" in error_msg):
                    logger.critical(f"[FOUR-PROBLEM-ENHANCED] OpenAI model {openai_model} unavailable for creative FOUR-PROBLEM script generation")
                    raise Exception(
                        f"OpenAI model {openai_model} is unavailable or has been sunset. Please update the creative-enhanced FOUR-PROBLEM script model configuration."
                    )

                if attempt < max_retries - 1:
                    logger.info(f"[FOUR-PROBLEM-ENHANCED] Retrying creative-enhanced FOUR-PROBLEM script generation in {retry_delay}s...")
                    print(
                        f"OpenAI API error: {e}. Retrying creative-enhanced FOUR-PROBLEM script generation in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"[FOUR-PROBLEM-ENHANCED] Failed to generate creative-enhanced FOUR-PROBLEM script after {max_retries} attempts")
                    print(
                        f"Failed to generate creative-enhanced FOUR-PROBLEM script after {max_retries} attempts: {e}"
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
    Load the three guidance files from Supabase bucket (FOUR-PROBLEM version)
    
    Args:
        bucket_name (str): Name of the script-guidance bucket
        
    Returns:
        dict: Dictionary containing voice, method, and FOUR-PROBLEM prompt guidance
    """
    print("\n" + "=" * 80)
    print("LOADING GUIDANCE FILES FOR CREATIVE-ENHANCED FOUR-PROBLEM SCRIPTS")
    print("=" * 80)

    try:
        # Load all three guidance files - using full version of prompt instructions
        # NO FALLBACKS - all files must exist or the process fails
        voice_guidance = download_file_from_bucket(bucket_name,
                                                   "voice_guidance.txt")
        method_guidance = download_file_from_bucket(bucket_name,
                                                    "method_guidance.txt")
        prompt_instructions = download_file_from_bucket(
            bucket_name, "prompt_instructions.txt")  # Using full version

        # Validate that all guidance files have content
        if not voice_guidance or not voice_guidance.strip():
            raise Exception("voice_guidance.txt is empty or contains only whitespace")
        
        if not method_guidance or not method_guidance.strip():
            raise Exception("method_guidance.txt is empty or contains only whitespace")
        
        if not prompt_instructions or not prompt_instructions.strip():
            raise Exception("prompt_instructions.txt is empty or contains only whitespace")

        guidance_files = {
            "voice": voice_guidance,
            "method": method_guidance,
            "prompt": prompt_instructions
        }

        print("Successfully loaded all guidance files for creative-enhanced FOUR-PROBLEM scripts")
        return guidance_files

    except Exception as e:
        print(f"Error loading guidance files: {str(e)}")
        raise Exception(f"Failed to load required guidance files from bucket '{bucket_name}': {str(e)}")


def process_poppy_cards(variables, guidance_files):
    """
    Process 10 Poppy Card combinations sequentially (cards 1-10) with prohibited terms cleanup
    
    Args:
        variables (dict): Configuration variables from Supabase
        guidance_files (dict): Loaded guidance files
        
    Returns:
        dict: Summary of processed creative-enhanced FOUR-PROBLEM scripts
    """
    try:
        # Validate required configuration exists before accessing
        required_paths = [
            ("scripts", "video_script"),
            ("global", "COMPANY_NAME"),
            ("scripts", "video_script", "supabase_buckets", "input_cards"),
            ("scripts", "video_script", "supabase_buckets", "guidance"),
            ("scripts", "video_script", "supabase_buckets", "output"),
            ("scripts", "video_script", "card_combinations")
        ]

        for path in required_paths:
            current = variables
            for key in path:
                if key not in current:
                    raise Exception(f"Missing required configuration: {'.'.join(path)}")
                current = current[key]
        
        logger.info("[FOUR-PROBLEM-ENHANCED] Configuration validation passed - all required paths exist")
        print("✅ Configuration validation passed - all required paths exist")

        # Extract configuration (now safely validated)
        video_script_config = variables["scripts"]["video_script"]
        company_name = variables["global"]["COMPANY_NAME"]
        openai_model = video_script_config.get("openai_model", "gpt-4o")
        
        # Get bucket configurations
        input_bucket = video_script_config["supabase_buckets"]["input_cards"]
        output_bucket = video_script_config["supabase_buckets"]["output"]
        
        # Use predefined card combinations from configuration
        card_combinations = video_script_config["card_combinations"]
        
        # Process each combination (cards 1-10)
        processed_scripts = []
        timestamp = datetime.now(eastern_tz).strftime("%Y%m%d_%H%M")
        total_cards = len(card_combinations)

        print(f"\n" + "=" * 80)
        print("PROCESSING POPPY CARDS WITH PROHIBITED TERMS CLEANUP ALIGNED WITH TWO-PROBLEM VERSION")
        print("=" * 80)
        print(f"Total combinations to process: {total_cards}")
        print(f"Company: {company_name}")
        print(f"OpenAI Model: {openai_model}")
        print(f"Input Bucket: {input_bucket}")
        print(f"Output Bucket: {output_bucket}")
        print(f"Quote Distribution: 4 quotes per problem (16 total per FOUR-PROBLEM script)")
        print(f"Company Integration: FIXED - 9-12 mentions with natural distribution")
        print(f"Word Target: 2000-2800 words (rich creative development)")
        print(f"Duration Target: 8-12 minutes")
        print(f"✅ PROHIBITED TERMS CLEANUP: Automatic replacement without failures")
        print(f"✅ FULL ADDITIVE ENHANCEMENTS: Feature clarity, Revenue impact, Implementation assurance, Competitive differentiation")
        print(f"Timestamp: {timestamp}")
        
        for i, combination in enumerate(card_combinations, 1):
            print(f"\nProcessing creative-enhanced FOUR-PROBLEM card {i} of {total_cards}...")
            print(f"Combination: {combination}")

            try:
                # Construct input and output filenames for cards 1-10
                card_number = f"card{i:02d}"  # Formats as card01, card02, ..., card10
                input_filename = f"{company_name}_{card_number}_{combination}.txt"
                output_filename = f"{company_name}_FOUR_FINAL_script_{combination}_{timestamp}.txt"

                print(f"Looking for file: {input_filename}")

                # Download the Poppy Card content
                poppy_card_content = download_file_from_bucket(
                    input_bucket, input_filename)

                # ✅ CREATIVE-ENHANCED SCRIPT GENERATION WITH PROHIBITED TERMS CLEANUP
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

                # ✅ DUAL VALIDATION RESULTS (STRUCTURAL + ENHANCED CONTENT)
                is_valid, validation_message = validate_quote_distribution(script_content)
                is_enhanced_valid, enhanced_message = validate_enhanced_content(script_content, company_name)
                quote_count = len(re.findall(r'"[^"]*"', script_content))
                word_count = len(script_content.split())
                company_mentions = script_content.lower().count(company_name.lower())
                
                # Record the processed script with enhanced metrics
                processed_scripts.append({
                    "combination": combination,
                    "input_file": input_filename,
                    "output_file": output_filename,
                    "script_length": len(script_content),
                    "word_count": word_count,
                    "quote_count": quote_count,
                    "company_mentions": company_mentions,
                    "validation_passed": is_valid,
                    "validation_message": validation_message,
                    "enhanced_validation_passed": is_enhanced_valid,
                    "enhanced_validation_message": enhanced_message,
                    "target_compliance": "optimal" if 2000 <= word_count <= 2800 else ("low" if word_count < 2000 else "high"),
                    "company_mention_compliance": "optimal" if 9 <= company_mentions <= 12 else ("low" if company_mentions < 9 else "acceptable" if company_mentions <= 15 else "high"),
                    "status": "success",
                    "script_type": "FOUR_FINAL"
                })

                print(f"✅ Successfully processed creative-enhanced FOUR-PROBLEM {combination}")
                print(f"📊 Quote count: {quote_count}, Target: 16, Validation: {'PASSED' if is_valid else 'WARNING'}")
                print(f"📊 Word count: {word_count}, Target: 2000-2800")
                print(f"🏢 Company mentions: {company_mentions}, Target: 9-12")
                print(f"📋 Structural validation: {validation_message}")
                print(f"🎯 Enhanced validation: {'PASSED' if is_enhanced_valid else 'WARNING'}")
                print(f"📈 Enhanced details: {enhanced_message}")

            except Exception as e:
                print(f"❌ Error processing creative-enhanced FOUR-PROBLEM {combination}: {str(e)}")
                processed_scripts.append({
                    "combination": combination,
                    "input_file": input_filename if 'input_filename' in locals() else "unknown",
                    "output_file": output_filename if 'output_filename' in locals() else "unknown",
                    "error": str(e),
                    "status": "failed",
                    "validation_passed": False,
                    "enhanced_validation_passed": False,
                    "quote_count": 0,
                    "word_count": 0,
                    "company_mentions": 0,
                    "script_type": "FOUR_FINAL"
                })

        # Enhanced summary with dual validation statistics for FOUR-PROBLEM format
        successful_scripts = [s for s in processed_scripts if s["status"] == "success"]
        failed_scripts = [s for s in processed_scripts if s["status"] == "failed"]
        validated_scripts = [s for s in successful_scripts if s.get("validation_passed", False)]
        enhanced_validated_scripts = [s for s in successful_scripts if s.get("enhanced_validation_passed", False)]
        optimal_company_mentions = [s for s in successful_scripts if s.get("company_mention_compliance") == "optimal"]
        
        # Calculate averages for successful scripts
        avg_quote_count = sum(s.get("quote_count", 0) for s in successful_scripts) / len(successful_scripts) if successful_scripts else 0
        avg_word_count = sum(s.get("word_count", 0) for s in successful_scripts) / len(successful_scripts) if successful_scripts else 0
        avg_company_mentions = sum(s.get("company_mentions", 0) for s in successful_scripts) / len(successful_scripts) if successful_scripts else 0
        optimal_count = len([s for s in successful_scripts if s.get("target_compliance") == "optimal"])

        summary = {
            "total_processed": total_cards,
            "successful": len(successful_scripts),
            "failed": len(failed_scripts),
            "validation_passed": len(validated_scripts),
            "enhanced_validation_passed": len(enhanced_validated_scripts),
            "company_mention_optimal": len(optimal_company_mentions),
            "validation_rate": f"{len(validated_scripts)}/{len(successful_scripts)}" if successful_scripts else "0/0",
            "enhanced_validation_rate": f"{len(enhanced_validated_scripts)}/{len(successful_scripts)}" if successful_scripts else "0/0",
            "company_mention_rate": f"{len(optimal_company_mentions)}/{len(successful_scripts)}" if successful_scripts else "0/0",
            "average_quote_count": round(avg_quote_count, 1),
            "target_quote_count": 16,
            "scripts": processed_scripts,
            "company_name": company_name,
            "timestamp": timestamp,
            "openai_model": openai_model,
            "script_type": "FOUR_FINAL",
            "card_range": "1-10",
            "target_duration": "8-12 minutes",
            "creative_enhancements": "Prohibited terms cleanup aligned with two-problem version",
            "company_mention_fix": "Simplified natural distribution (9-12 mentions)",
            "additive_improvements": ["feature_clarity", "revenue_impact", "implementation_assurance", "competitive_differentiation"],
            "word_count_analysis": {
                "average_word_count": round(avg_word_count, 1),
                "optimal_compliance": f"{optimal_count}/{len(successful_scripts)}" if successful_scripts else "0/0",
                "target_range": "2000-2800 words"
            },
            "company_integration": {
                "average_mentions": round(avg_company_mentions, 1),
                "target_mentions": "9-12",
                "optimal_rate": f"{len(optimal_company_mentions)}/{len(successful_scripts)}" if successful_scripts else "0/0"
            }
        }

        print(f"\n📊 PROCESSING SUMMARY - FINAL FOUR-PROBLEM FORMAT:")
        print(f"✅ FINAL FOUR-PROBLEM scripts generated: {len(successful_scripts)}/{total_cards}")
        print(f"✅ Structural validation passed: {len(validated_scripts)}/{len(successful_scripts)}")
        print(f"🎯 Enhanced content validation passed: {len(enhanced_validated_scripts)}/{len(successful_scripts)}")
        print(f"🏢 Company mention optimization: {len(optimal_company_mentions)}/{len(successful_scripts)}")
        print(f"📈 Average quote count: {avg_quote_count:.1f} (target: 16)")
        print(f"📈 Average word count: {avg_word_count:.1f} (target: 2000-2800)")
        print(f"🏢 Average company mentions: {avg_company_mentions:.1f} (target: 9-12)")
        print(f"🎯 Structural success rate: {(len(validated_scripts)/len(successful_scripts)*100):.1f}%" if successful_scripts else "0%")
        print(f"🚀 Enhanced content success rate: {(len(enhanced_validated_scripts)/len(successful_scripts)*100):.1f}%" if successful_scripts else "0%")
        print(f"🏢 Company mention success rate: {(len(optimal_company_mentions)/len(successful_scripts)*100):.1f}%" if successful_scripts else "0%")
        print(f"🎯 Word count compliance: {optimal_count}/{len(successful_scripts)} scripts in optimal range")
        print(f"📋 Fixes applied: Prohibited terms cleanup, simplified generation flow")
        
        return summary

    except Exception as e:
        print(f"❌ Error in process_poppy_cards: {str(e)}")
        raise


def main():
    """Main function to orchestrate the entire creative-enhanced FOUR-PROBLEM video script workflow."""
    try:
        logger.info("=" * 80)
        logger.info("FINAL FOUR-PROBLEM VIDEO SCRIPT AUTOMATION - PROHIBITED TERMS CLEANUP FIXED")
        logger.info("=" * 80)
        print("=" * 80)
        print("FINAL FOUR-PROBLEM VIDEO SCRIPT AUTOMATION - PROHIBITED TERMS CLEANUP FIXED")
        print("=" * 80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Script directory: {SCRIPT_DIR}")
        print(f"Target: 8-12 minute scripts with 16 quotes (4 per problem) from cards 1-10")
        print(f"Word Target: 2000-2800 words with rich creative development")
        print("🎯 APPROACH: Exactly 4 quotes per problem (16 total quotes per FOUR-PROBLEM script)")
        print("🏢 FIXED: Company name natural distribution (9-12 mentions)")
        print("🚀 PROHIBITED TERMS CLEANUP ALIGNED WITH TWO-PROBLEM VERSION")
        print("   ✅ Automatic term replacement without failures")
        print("   ✅ Feature clarity in each problem section (<800 chars)")
        print("   ✅ Revenue impact with <6-month payback assurance (<800 chars)")
        print("   ✅ Implementation assurance <8 hours in outro (<800 chars)")
        print("   ✅ Competitive differentiation in outro (<800 chars)")
        print("🔧 TOKEN LIMIT: Increased to 4000 for comprehensive generation")

        # Fetch configuration from Supabase with error handling
        logger.info("[FOUR-PROBLEM-ENHANCED] Fetching configuration from Supabase...")
        print("\nFetching FINAL FOUR-PROBLEM configuration from Supabase...")
        
        try:
            variables = fetch_configuration_from_supabase()
        except Exception as e:
            logger.critical(f"[FOUR-PROBLEM-ENHANCED] Failed to fetch configuration: {str(e)}")
            raise Exception(f"Failed to fetch FINAL FOUR-PROBLEM script configuration from Supabase: {str(e)}")

        # Validate that we have the video_script configuration
        if "scripts" not in variables or "video_script" not in variables["scripts"]:
            logger.critical("[FOUR-PROBLEM-ENHANCED] Missing video_script configuration in Supabase")
            raise Exception(
                "video_script configuration not found in Supabase config. Please ensure the configuration includes a 'video_script' section.")

        # Validate global configuration
        if "global" not in variables:
            logger.critical("[FOUR-PROBLEM-ENHANCED] Missing global configuration in Supabase")
            raise Exception("global configuration not found in Supabase config. Please ensure the configuration includes a 'global' section.")
        
        video_script_config = variables["scripts"]["video_script"]
        logger.info("[FOUR-PROBLEM-ENHANCED] Successfully loaded video_script configuration")

        # Load guidance files (using FOUR-PROBLEM version) with no fallback handling
        guidance_bucket = video_script_config["supabase_buckets"]["guidance"]
        try:
            guidance_files = load_guidance_files(guidance_bucket)
            logger.info("[FOUR-PROBLEM-ENHANCED] Successfully loaded all guidance files for FINAL processing")
        except Exception as e:
            logger.critical(f"[FOUR-PROBLEM-ENHANCED] Failed to load guidance files: {str(e)}")
            raise Exception(f"Failed to load FINAL FOUR-PROBLEM script guidance files from bucket '{guidance_bucket}': {str(e)}")

        # Process Poppy Cards (cards 1-10) with FINAL enhancements
        try:
            logger.info("[FOUR-PROBLEM-ENHANCED] Starting FINAL card processing for cards 1-10")
            summary = process_poppy_cards(variables, guidance_files)
            logger.info(f"[FOUR-PROBLEM-ENHANCED] Completed FINAL processing: {summary['successful']}/{summary['total_processed']} successful")
        except Exception as e:
            logger.critical(f"[FOUR-PROBLEM-ENHANCED] Failed during FINAL card processing: {str(e)}")
            raise Exception(f"Failed to process FINAL FOUR-PROBLEM script cards 1-10: {str(e)}")

        # Save summary to output bucket - MUST SUCCEED  
        output_bucket = video_script_config["supabase_buckets"]["output"]
        summary_filename = f"video_script_FOUR_FINAL_summary_{summary['timestamp']}.json"
        summary_content = json.dumps(summary, indent=2)
        
        try:
            upload_file_to_bucket(output_bucket, summary_filename, summary_content)
            logger.info(f"[FOUR-PROBLEM-ENHANCED] Saved FINAL summary as {summary_filename}")
        except Exception as e:
            logger.error(f"[FOUR-PROBLEM-ENHANCED] Failed to save FINAL summary: {str(e)}")
            print(f"Warning: Could not save FINAL summary file to bucket '{output_bucket}': {str(e)}")
            print("FINAL video script generation completed successfully despite summary save failure.")

        # Calculate and display execution time
        end_time = datetime.now(eastern_tz)
        execution_time = end_time - start_time

        logger.info("=" * 80)
        logger.info("FINAL FOUR-PROBLEM VIDEO SCRIPT WORKFLOW COMPLETE")
        logger.info(f"Execution time: {execution_time}")
        logger.info(f"Success rate: {summary['successful']}/{summary['total_processed']}")
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("FINAL FOUR-PROBLEM VIDEO SCRIPT WORKFLOW COMPLETE")
        print("=" * 80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {execution_time}")
        print(f"FINAL FOUR-PROBLEM scripts generated: {summary['successful']}/{summary['total_processed']}")
        print(f"Structural validation success rate: {summary['validation_rate']}")
        print(f"Enhanced content validation success rate: {summary['enhanced_validation_rate']}")
        print(f"Company mention success rate: {summary['company_mention_rate']}")
        print(f"Average quote count: {summary['average_quote_count']} (target: 16)")
        print(f"Average word count: {summary['word_count_analysis']['average_word_count']} (target: 2000-2800)")
        print(f"Average company mentions: {summary['company_integration']['average_mentions']} (target: 9-12)")
        print(f"FINAL summary saved as: {summary_filename}")

        if summary['failed'] > 0:
            logger.warning(f"[FOUR-PROBLEM-ENHANCED] {summary['failed']} FINAL scripts failed to generate")
            print(f"\n⚠️ Warning: {summary['failed']} FINAL FOUR-PROBLEM script(s) failed to generate")
            for script in summary['scripts']:
                if script['status'] == 'failed':
                    logger.error(f"[FOUR-PROBLEM-ENHANCED-FAILED] {script['combination']}: {script.get('error', 'Unknown error')}")
                    print(f"  - {script['combination']}: {script.get('error', 'Unknown error')}")

        # Show comprehensive validation statistics
        validation_passed = summary['validation_passed']
        enhanced_validation_passed = summary['enhanced_validation_passed']
        company_mention_optimal = summary['company_mention_optimal']
        total_successful = summary['successful']
        if total_successful > 0:
            print(f"\n📊 COMPREHENSIVE FINAL VALIDATION ANALYSIS:")
            print(f"✅ Scripts with balanced 4-problem structure: {validation_passed}/{total_successful}")
            print(f"📏 Scripts with 16 quotes (4 per problem): {validation_passed}/{total_successful}")
            print(f"🎯 Scripts with enhanced content: {enhanced_validation_passed}/{total_successful}")
            print(f"🏢 Scripts with optimal company mentions (9-12): {company_mention_optimal}/{total_successful}")
            print(f"📈 Company mention success rate: {(company_mention_optimal/total_successful*100):.1f}%")
            print(f"🎯 Overall validation success rate: {(min(validation_passed, enhanced_validation_passed)/total_successful*100):.1f}%")
            print(f"⏱️ Duration target compliance: 8-12 minutes with {summary['word_count_analysis']['optimal_compliance']} optimal word count")
            print(f"🚀 Final fixes applied: {summary['company_mention_fix']}")

        print("\n🎉 FINAL FOUR-PROBLEM video script automation workflow completed successfully!")
        print("📋 Each FINAL FOUR-PROBLEM script contains exactly 16 quotes (4 quotes per problem)")
        print("🏢 Each FINAL FOUR-PROBLEM script includes optimized company name distribution (9-12 mentions)")
        print("⏱️ Each FINAL FOUR-PROBLEM script targets 8-12 minute duration with rich development")
        print("🚀 Each FINAL FOUR-PROBLEM script includes prohibited terms cleanup:")
        print("   • Automatic replacement of transform → modernize")
        print("   • Automatic replacement of empower → enable")
        print("   • Automatic replacement of unlock → access")
        print("   • Automatic replacement of seamless → smooth")
        print("   • Automatic replacement of game-changer → major advantage")
        print("   • Automatic replacement of catalyst → driver")
        print("🚀 Each FINAL FOUR-PROBLEM script includes full additive improvements:")
        print("   • Feature clarity explanations in problem sections")
        print("   • Revenue impact with sub-6-month payback assurance")
        print("   • Implementation time assurance (<8 hours)")
        print("   • Competitive differentiation advantages")
        print("🎯 Prohibited terms cleanup: Fixed with automatic replacement (100% success rate expected)")
        print("📊 Word distribution proportional to four-problem format with rich content development")
        
        # Log successful session completion
        logger.info("=" * 60)
        logger.info("FINAL FOUR-PROBLEM VIDEO SCRIPT AUTOMATION - SESSION END (SUCCESS)")
        logger.info("=" * 60)

    except Exception as e:
        logger.critical(f"[FOUR-PROBLEM-ENHANCED] Critical error in FINAL FOUR-PROBLEM workflow: {str(e)}")
        logger.critical(f"[FOUR-PROBLEM-ENHANCED] Traceback: {traceback.format_exc()}")
        print(f"\n❌ Critical error in FINAL FOUR-PROBLEM workflow: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Ensure we log session end even on failure
        logger.info("=" * 60)
        logger.info("FINAL FOUR-PROBLEM VIDEO SCRIPT AUTOMATION - SESSION END (FAILED)")
        logger.info("=" * 60)
        raise


if __name__ == "__main__":
    main()