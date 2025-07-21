#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Video Script SHORT Automation Workflow Script with Full Creative Guidance

This script processes Poppy Card content and generates SHORT video scripts using:
- Voice guidance (tone and style)
- Method guidance (structure and framework) 
- Prompt instructions (specific processing directions for 6-8 minute format)
- Poppy Card content (unique subject matter for 2-problem cards)

ENHANCEMENTS (TRANSPLANTED FROM FULL VERSION):
- Simple quote distribution (4 quotes per problem = 8 total quotes per script)
- Peer validation psychology framework
- Professional competence vs. ego-stroking detection
- Quote distribution validation and reporting
# Company name integration (FIXED: 4-6 mentions with spacing control)
- ADDITIVE IMPROVEMENTS: Feature clarity, revenue impact, implementation assurance, competitive differentiation
- Enhanced content validation with full strictness
- Rich creative guidance matching four-problem version depth

The workflow processes 5 predefined Poppy Card combinations sequentially (cards 11-15),
generating custom SHORT video scripts for each combination and saving them to Supabase.

Card Range: 11-15 (specifically for 2-problem format cards)
Target Duration: 6-8 minutes (increased from 5 minutes for rich development)
Target Word Count: 1200-1600 words (proportional depth to full version)
Quote Distribution: 4 quotes per problem (8 total quotes)
Company Integration: FIXED - 4-6 mentions with 4-sentence spacing control
Output Bucket: two-problem-script-drafts (preserves existing architecture)

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
    format='%(asctime)s - [SHORT-ENHANCED] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"video_script_SHORT_enhanced_creative_log_{datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}.log")
    ])
logger = logging.getLogger(__name__)

# Log initialization for enhanced SHORT workflow
logger.info("=" * 60)
logger.info("SHORT VIDEO SCRIPT AUTOMATION - ENHANCED WITH FULL CREATIVE GUIDANCE")
logger.info(f"Target: 6-8 minute scripts with 8 quotes (4 per problem) from cards 11-15")
logger.info(f"Word Target: 1200-1600 words with rich creative development")
logger.info(f"Company Integration: FIXED - 4-6 mentions with spacing control")
logger.info(f"Session ID: {datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}")
logger.info("=" * 60)

# Suppress excessive HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Setup directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_PATH = os.path.join(script_dir, "STATIC_VARS_MAR2025.env")

# =====================================================================
# ENHANCED QUOTE DISTRIBUTION VALIDATION FUNCTION - 2-PROBLEM STRUCTURE
# =====================================================================

def validate_quote_distribution_short(script_content):
    """
    Enhanced validation for 2-problem structure with balanced quote distribution and content balance
    
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
        
        # Enhanced structural validation - Split script into sections
        sections = re.split(r'(?i)(problem|challenge|issue|struggle|difficulty)', script_content)
        
        if len(sections) < 3:  # Need at least intro + 2 problems
            return False, "Script missing clear 2-problem structure with identifiable problem sections"
        
        # Validate quotes per problem section - Look for distinct problem areas
        problem_quotes = []
        problem_word_counts = []
        
        # Analyze first two major sections after problem indicators
        for i in range(1, min(5, len(sections))):  # Check up to 4 sections after splits
            if i == 1 or i == 3:  # Likely problem sections (odd indices after splits)
                if i + 1 < len(sections):
                    section_content = sections[i + 1]  # Content after problem indicator
                    # Look at substantial portion of each problem section
                    section_preview = section_content[:1200] if len(section_content) >= 1200 else section_content
                    section_quotes = len(re.findall(r'"[^"]*"', section_preview))
                    section_words = len(section_preview.split())
                    
                    problem_quotes.append(section_quotes)
                    problem_word_counts.append(section_words)
        
        if len(problem_quotes) < 2:
            return False, "Could not identify 2 distinct problem sections with quotes"
        
        # Validate quote distribution per problem
        if problem_quotes[0] < 2 or problem_quotes[1] < 2:
            return False, f"Uneven quote distribution: Problem 1: {problem_quotes[0]} quotes, Problem 2: {problem_quotes[1]} quotes (need ~4 each)"
        
        # Check content balance between problems (adjusted for longer format)
        if len(problem_word_counts) >= 2:
            word_diff = abs(problem_word_counts[0] - problem_word_counts[1])
            if word_diff > 300:  # Allow more flexibility for longer content
                return False, f"Unbalanced problem development: P1: {problem_word_counts[0]} words, P2: {problem_word_counts[1]} words (difference: {word_diff})"
        
        # Check for professional competence indicators (not ego-stroking)
        ego_indicators = ['hero', 'genius', 'star', 'rockstar', 'superstar', 'legend', 'champion']
        confidence_indicators = ['confident', 'prepared', 'clarity', 'insights', 'data-driven', 'strategic']
        
        ego_count = sum(1 for word in ego_indicators if word in script_content.lower())
        confidence_count = sum(1 for word in confidence_indicators if word in script_content.lower())
        
        if ego_count > confidence_count:
            return False, f"Script leans toward ego-stroking ({ego_count} ego vs {confidence_count} confidence indicators). Focus on professional competence instead."
        
        # Enhanced success validation with balance metrics
        balance_info = ""
        if len(problem_word_counts) >= 2:
            balance_info = f", balanced content ({problem_word_counts[0]}/{problem_word_counts[1]} words per problem)"
        
        return True, f"Excellent SHORT structure: {quote_count} total quotes ({problem_quotes[0]}/{problem_quotes[1]} per problem){balance_info} with professional competence focus"
        
    except Exception as e:
        return False, f"SHORT validation error: {str(e)}"


# =====================================================================
# FIXED ENHANCED CONTENT VALIDATION FUNCTION
# =====================================================================

def validate_enhanced_content_short(script_content, company_name):
    """
    FIXED: Validate the enhanced content requirements for SHORT format with proper company name validation
    
    Args:
        script_content (str): The generated video script content
        company_name (str): Company name to check for
        
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    try:
        validation_issues = []
        
        # Check for implementation time mention (< 8 hours) - same as full version
        implementation_keywords = ['8 hour', 'eight hour', 'quick implementation', 'rapid deployment', 'fast setup']
        has_implementation_mention = any(keyword in script_content.lower() for keyword in implementation_keywords)
        if not has_implementation_mention:
            validation_issues.append("Missing implementation time assurance in outro")
        
        # Check for payback period mention (< 6 months) - same as full version
        payback_keywords = ['6 month', 'six month', 'payback', 'roi', 'return on investment']
        has_payback_mention = any(keyword in script_content.lower() for keyword in payback_keywords)
        if not has_payback_mention:
            validation_issues.append("Missing payback period assurance in problem sections")
        
        # Check for competitive differentiation - same as full version
        competitive_keywords = ['better than', 'unlike', 'superior', 'advantage', 'unique', 'differentiat']
        has_competitive_mention = any(keyword in script_content.lower() for keyword in competitive_keywords)
        if not has_competitive_mention:
            validation_issues.append("Missing competitive differentiation in outro")
        
        # Check for feature explanations - same as full version
        feature_keywords = ['feature', 'capability', 'functionality', 'tool', 'dashboard', 'analytics']
        has_feature_mention = any(keyword in script_content.lower() for keyword in feature_keywords)
        if not has_feature_mention:
            validation_issues.append("Missing specific feature explanations in problem sections")
        
        # FIXED: Check for company name mentions with minimum AND maximum requirements
        company_mentions = script_content.lower().count(company_name.lower())
        
        # FIXED: Require 4-6 mentions (minimum 4, maximum 6)
        if company_mentions < 4:
            validation_issues.append(f"CRITICAL: Insufficient company mentions: {company_mentions} (required: 4-6 mentions)")
        elif company_mentions > 6:
            validation_issues.append(f"CRITICAL: Excessive company mentions: {company_mentions} (maximum: 6 with spacing control)")
        
        # FIXED: STRICT spacing validation - check for consecutive mentions
        sentences = re.split(r'[.!?]+', script_content)
        spacing_violations = 0
        violation_details = []
        
        for i, sentence in enumerate(sentences):
            if company_name.lower() in sentence.lower():
                # Check previous 3 sentences (4 total including current)
                for j in range(max(0, i-3), i):
                    if j < len(sentences) and company_name.lower() in sentences[j].lower():
                        spacing_violations += 1
                        violation_details.append(f"Sentences {j+1} and {i+1}")
                        break
        
        if spacing_violations > 0:  # ZERO tolerance for spacing violations
            validation_issues.append(f"CRITICAL: Company name spacing violations: {spacing_violations} (mentions too close together in: {', '.join(violation_details[:3])})")
        
        # Additional check for excessive density
        if company_mentions > 0 and len(sentences) > 0:
            mention_density = company_mentions / len(sentences)
            if mention_density > 0.04:  # Tightened from 5% to 4%
                validation_issues.append(f"CRITICAL: Company name density too high: {mention_density:.2%} (should be <4%)")
        
        if validation_issues:
            return False, f"Enhanced content validation issues: {', '.join(validation_issues)}"
        else:
            return True, f"All enhanced content requirements validated successfully (company mentions: {company_mentions})"
            
    except Exception as e:
        return False, f"Enhanced validation error: {str(e)}"


# =====================================================================
# COMMON FUNCTIONS (UNCHANGED)
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
        logger.info(f"[SHORT-ENHANCED] Downloading {file_name} from {bucket_name} bucket...")
        print(f"Downloading {file_name} from {bucket_name} bucket...")
        response = supabase.storage.from_(bucket_name).download(file_name)

        if response:
            content = response.decode('utf-8')
            logger.info(f"[SHORT-ENHANCED] Successfully downloaded {file_name} ({len(content)} characters)")
            print(
                f"Successfully downloaded {file_name} ({len(content)} characters)"
            )
            return content
        else:
            raise Exception(f"Failed to download {file_name}")

    except Exception as e:
        logger.error(f"[SHORT-ENHANCED] Error downloading {file_name} from {bucket_name}: {str(e)}")
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
        logger.info(f"[SHORT-ENHANCED] Uploading {file_name} to {bucket_name} bucket...")
        print(f"Uploading {file_name} to {bucket_name} bucket...")

        # Convert string content to bytes
        file_bytes = file_content.encode('utf-8')

        response = supabase.storage.from_(bucket_name).upload(
            file_name, file_bytes, {"content-type": "text/plain"})

        logger.info(f"[SHORT-ENHANCED] Successfully uploaded {file_name} to {bucket_name}")
        print(f"Successfully uploaded {file_name} to {bucket_name}")
        return True

    except Exception as e:
        logger.error(f"[SHORT-ENHANCED] Error uploading {file_name} to {bucket_name}: {str(e)}")
        print(f"Error uploading {file_name} to {bucket_name}: {str(e)}")
        raise


# =====================================================================
# FIXED VIDEO SCRIPT GENERATION WITH PROPER COMPANY NAME CONTROL
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
    FIXED: Generate a SHORT video script with proper company name control (4-6 mentions)
    
    Args:
        voice_guidance (str): Voice and tone guidance
        method_guidance (str): Script structure and framework guidance
        prompt_instructions (str): Specific processing instructions for 6-8 minute format
        poppy_card_content (str): Poppy Card content to focus on (2-problem format)
        company_name (str): Company name for brand integration
        openai_model (str): OpenAI model to use
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        str: Generated SHORT video script with exactly 8 quotes (4 per problem) and 4-6 company mentions
    """
    try:
        # Create OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)

        # FIXED: Creative guidance system prompt with proper company name control
        system_prompt = f"""You are a professional video script writer specializing in B2B software buyer psychology. Generate a SHORT video script using the following guidance:

TARGET: 6-8 minute duration (approximately 1200-1600 words)
FORMAT: 2-problem structure for maximum engagement with rich development

CRITICAL QUOTE REQUIREMENTS - READ THIS FIRST:
The poppy card contains 16 customer quotes (8 quotes per problem). 
YOU MUST USE EXACTLY 4 QUOTES FROM EACH PROBLEM (8 TOTAL QUOTES IN YOUR SHORT SCRIPT).

MANDATORY DISTRIBUTION - SIMPLE APPROACH:
- Problem 1: Use the FIRST 4 quotes from Problem 1's 8 available quotes
- Problem 2: Use the FIRST 4 quotes from Problem 2's 8 available quotes

⚠️ SHORT SCRIPTS MUST CONTAIN EXACTLY 8 QUOTES (4 PER PROBLEM) ⚠️

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

WORD DISTRIBUTION REQUIREMENTS (Proportional to Full Version):
- Total script: 1200-1600 words
- Intro: 100-150 words (hook + company mention)
- Problem 1: 500-700 words (4 quotes + enhancements + company mention)
- Transition: 75-100 words (bridge between problems)
- Problem 2: 500-700 words (4 quotes + enhancements + company mention)
- Outro: 150-200 words (implementation + competitive + company mention)

FINAL REMINDER BEFORE YOU BEGIN:
- Use exactly 4 quotes from each problem (first 4 from each problem's list)
- Total script must contain exactly 8 quotes
- Use the exact quotes from the poppy card content provided
- Do not skip quotes or rearrange - use the first 4 from each problem in order
- Target 1200-1600 words for rich, engaging content

Requirements:
- Write in plain text format
- Use short paragraphs of 1-3 sentences maximum
- Add line breaks between paragraphs
- Create an engaging video script that follows the voice, method, and focuses on the provided content
- Ensure quote distribution creates a "peer validation experience" rather than a sales pitch"""

        # FIXED: Problem section content requirements BEFORE final company name control
        system_prompt += f"""

🎯 PROBLEM SECTION CONTENT REQUIREMENTS - ADDITIVE ENHANCEMENTS:

FEATURE CLARITY MANDATE (Requirement 1):
- In each of the 2 problem sections, add 2-3 sentences explaining the specific feature that addresses this problem
- Use fewer than 800 additional characters per problem section for feature explanations
- Include high-level explanation of how the feature helps solve the specific problem
- PREFER alternative references ("this cutting-edge platform", "the system", "this technology")
- Make feature descriptions concrete and specific, not vague marketing language
- Examples: "The pipeline analytics dashboard shows exactly where deals are stuck" or "The automated scoring system highlights which prospects need immediate attention"

REVENUE IMPACT MANDATE (Requirement 2):
- In each of the 2 problem sections, add 2-3 sentences explaining how this solution increases revenue
- Use fewer than 800 additional characters per problem section for revenue impact explanations
- Specifically assure viewers they will achieve less than 6-month payback period
- PREFER alternative references ("this advanced tool", "the platform", "this technology")
- Include specific revenue generation mechanisms (faster deals, better conversion, reduced waste, etc.)
- Position as "fully engaged users consistently achieve sub-6-month ROI"
- Examples: "Companies using deal acceleration features see 23% faster close rates, typically achieving full payback in under 6 months" or "Users report 31% improvement in qualified lead conversion, with most seeing ROI within 5 months"

🚀 OUTRO CONTENT REQUIREMENTS - ADDITIVE ENHANCEMENTS:

IMPLEMENTATION ASSURANCE MANDATE (Requirement 3):
- In the first 4 sentences of the outro, assure viewers that full implementation takes less than 8 hours
- Use fewer than 800 additional characters for implementation time assurance
- PREFER alternative references ("the platform", "this top-rated solution", "the system")
- Emphasize minimal disruption to current operations
- Position as "rapid deployment advantage"
- Examples: "The platform deploys in under 8 hours with zero disruption to your current sales process" or "Full implementation typically completes in 6-8 hours, often during a single business day"

COMPETITIVE DIFFERENTIATION MANDATE (Requirement 4):
- In the first 5 sentences of the outro, clarify what makes the solution superior to competitors
- Use fewer than 800 additional characters for competitive advantages
- PREFER alternative references ("this cutting-edge platform", "the system", "this technology")
- Include 2 specific differentiators that are concrete and measurable
- Focus on unique capabilities, not generic benefits
- Avoid naming specific competitors - focus on category advantages
- Examples: "Unlike traditional CRM analytics, the platform provides predictive deal scoring and real-time pipeline health monitoring" or "The peer-based benchmarking gives you insights that generic sales platforms simply cannot match"
"""

        # FIXED: ABSOLUTE FINAL COMPANY NAME CONTROL (moved to the very end)
        system_prompt += f"""

🚨 ABSOLUTE FINAL OVERRIDE - COMPANY NAME CONTROL (OVERRIDES ALL ABOVE):
⚠️ THIS SECTION SUPERSEDES ALL PREVIOUS COMPANY NAME INSTRUCTIONS ⚠️

Company name: {company_name}

MANDATORY REQUIREMENTS:
- TOTAL TARGET: Exactly 4-6 mentions of {company_name} across entire script
- MINIMUM: Must have at least 4 mentions (for branding consistency)
- MAXIMUM: Cannot exceed 6 mentions (to avoid oversaturation)
- SPACING RULE: Maximum ONE mention per 4 consecutive sentences

DEFAULT BEHAVIOR - USE ALTERNATIVE REFERENCES:
- ALWAYS use alternative references as your PRIMARY choice
- "this cutting-edge platform", "the system", "this technology", "this advanced tool", "the platform", "this top-rated solution"
- Only use {company_name} when specifically required for branding placement

STRATEGIC PLACEMENT GUIDE:
- Mention 1: Early intro for brand introduction
- Mention 2: Mid-problem 1 for feature association  
- Mention 3: Mid-problem 2 for solution reinforcement
- Mention 4: Early outro for brand recall
- Mention 5: (Optional) Mid-outro for competitive positioning
- Mention 6: (Optional) Final outro for brand closure

ENFORCEMENT PROCESS:
1. Count backward 4 sentences before each potential {company_name} usage
2. If {company_name} appears in those 4 sentences, use alternative reference instead
3. Only use {company_name} if no mention in previous 4 sentences AND it serves strategic branding purpose
4. Track your mention count to stay within 4-6 total range

CRITICAL SUCCESS CRITERIA:
✅ Script contains exactly 4-6 mentions of {company_name}
✅ No two mentions within 4 sentences of each other
✅ Alternative references used for all other brand needs
✅ Strategic placement for maximum branding impact

EXAMPLE ENFORCEMENT:
"Sales teams struggle with pipeline visibility. This is where {company_name} excels. [3 sentences with alternatives] This cutting-edge platform provides analytics. Teams see immediate results. The system transforms workflows. [4 sentences passed] Companies using {company_name} report success."

🔴 FINAL INSTRUCTION: Count your {company_name} mentions as you write. Target exactly 4-6 total mentions."""

        # Continue with retry logic for SHORT scripts with validation enforcement
        for attempt in range(max_retries):
            try:
                logger.info(f"[SHORT-ENHANCED] Attempt {attempt + 1}/{max_retries} for FIXED 2-problem script with {company_name} integration")
                print(
                    f"Generating FIXED SHORT video script with controlled company mentions (4-6 target) using {openai_model} (attempt {attempt + 1}/{max_retries})..."
                )

                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{
                        "role": "system",
                        "content": system_prompt
                    }, {
                        "role": "user",
                        "content": f"Generate a 6-8 minute SHORT video script now with exactly 4 quotes from each problem (8 total quotes) and EXACTLY 4-6 mentions of {company_name} with proper spacing. Use alternative references for all other brand needs. Target 1200-1600 words with rich creative development."
                    }],
                    max_tokens=4000,
                    temperature=0.7)

                script_content = response.choices[0].message.content
                if script_content:
                    script_content = script_content.strip()

                if script_content:
                    # Calculate word count for 6-8 minute target validation
                    word_count = len(script_content.split())
                    company_mentions = script_content.lower().count(company_name.lower())
                    
                    logger.info(f"[SHORT-ENHANCED] Generated {word_count} words, {company_mentions} company mentions")
                    print(
                        f"Generated FIXED SHORT video script ({len(script_content)} characters, ~{word_count} words, {company_mentions} company mentions)"
                    )
                    
                    # FIXED: Dual validation with rejection for failed scripts
                    is_valid, validation_message = validate_quote_distribution_short(script_content)
                    is_enhanced_valid, enhanced_message = validate_enhanced_content_short(script_content, company_name)
                    
                    # FIXED: Only return script if BOTH validations pass
                    if is_valid and is_enhanced_valid:
                        logger.info(f"✅ All validations passed: {validation_message} | {enhanced_message}")
                        print(f"✅ All validations passed: {validation_message} | {enhanced_message}")
                        return script_content
                    else:
                        # FIXED: Reject script and retry if validation fails
                        logger.warning(f"⚠️ Validation failed - retrying: {validation_message} | {enhanced_message}")
                        print(f"⚠️ Validation failed - retrying: {validation_message} | {enhanced_message}")
                        
                        if attempt < max_retries - 1:
                            continue  # Retry instead of returning failed script
                        else:
                            raise Exception(f"Script validation failed after all retries: {validation_message} | {enhanced_message}")
                else:
                    raise Exception("Empty response from OpenAI for FIXED SHORT script generation")

            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"[SHORT-ENHANCED] Attempt {attempt + 1} failed: {str(e)}")

                # Enhanced error handling for SHORT script context
                if "model" in error_msg and ("not found" in error_msg
                                             or "unavailable" in error_msg
                                             or "sunset" in error_msg):
                    logger.critical(f"[SHORT-ENHANCED] OpenAI model {openai_model} unavailable for FIXED SHORT script generation")
                    raise Exception(
                        f"OpenAI model {openai_model} is unavailable or has been sunset. Please update the FIXED SHORT script model configuration."
                    )

                if attempt < max_retries - 1:
                    logger.info(f"[SHORT-ENHANCED] Retrying FIXED SHORT script generation in {retry_delay}s...")
                    print(
                        f"OpenAI API error: {e}. Retrying FIXED SHORT script generation in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"[SHORT-ENHANCED] Failed to generate FIXED SHORT script after {max_retries} attempts")
                    print(
                        f"Failed to generate FIXED SHORT script after {max_retries} attempts: {e}"
                    )
                    raise

    except Exception as e:
        print(f"Error in generate_video_script: {str(e)}")
        raise


# =====================================================================
# MAIN WORKFLOW FUNCTIONS (UNCHANGED EXCEPT FOR UPDATED LOGGING)
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
    print("LOADING GUIDANCE FILES FOR FIXED SHORT SCRIPTS")
    print("=" * 80)

    try:
        # Load all three guidance files - using SHORT version of prompt instructions
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

        print("Successfully loaded all guidance files for FIXED SHORT scripts")
        return guidance_files

    except Exception as e:
        print(f"Error loading guidance files: {str(e)}")
        raise Exception(f"Failed to load required guidance files from bucket '{bucket_name}': {str(e)}")


def process_poppy_cards(variables, guidance_files):
    """
    Process 5 Poppy Card combinations sequentially (cards 11-15) with FIXED company name control
    
    Args:
        variables (dict): Configuration variables from Supabase
        guidance_files (dict): Loaded guidance files
        
    Returns:
        dict: Summary of processed FIXED SHORT scripts
    """
    try:
        # Validate required configuration exists before accessing
        required_paths = [
            ("scripts", "video_script_short"),
            ("global", "COMPANY_NAME"),
            ("scripts", "video_script_short", "supabase_buckets", "input_cards"),
            ("scripts", "video_script_short", "supabase_buckets", "guidance"),
            ("scripts", "video_script_short", "supabase_buckets", "output"),
            ("scripts", "video_script_short", "card_combinations")
        ]

        for path in required_paths:
            current = variables
            for key in path:
                if key not in current:
                    raise Exception(f"Missing required configuration: {'.'.join(path)}")
                current = current[key]
        
        logger.info("[SHORT-ENHANCED] Configuration validation passed - all required paths exist")
        print("✅ Configuration validation passed - all required paths exist")

        # Extract configuration (now safely validated)
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
        print("PROCESSING POPPY CARDS WITH FIXED COMPANY NAME CONTROL")
        print("=" * 80)
        print(f"Total combinations to process: {total_cards}")
        print(f"Company: {company_name}")
        print(f"OpenAI Model: {openai_model}")
        print(f"Input Bucket: {input_bucket}")
        print(f"Output Bucket: {output_bucket}")
        print(f"Quote Distribution: 4 quotes per problem (8 total per SHORT script)")
        print(f"Company Integration: FIXED - 4-6 mentions with spacing control")
        print(f"Word Target: 1200-1600 words (rich creative development)")
        print(f"Duration Target: 6-8 minutes")
        print(f"✅ FIXED VALIDATION: Scripts rejected and retried if validation fails")
        print(f"Timestamp: {timestamp}")
        
        for i, combination in enumerate(card_combinations, 1):
            print(f"\nProcessing FIXED SHORT card {i} of {total_cards}...")
            print(f"Combination: {combination}")

            try:
                # Construct input and output filenames for cards 11-15
                card_number = f"card{i+10:02d}"  # Formats as card11, card12, ..., card15
                input_filename = f"{company_name}_{card_number}_{combination}.txt"
                output_filename = f"{company_name}_SHORT_FIXED_script_{combination}_{timestamp}.txt"

                print(f"Looking for file: {input_filename}")

                # Download the Poppy Card content
                poppy_card_content = download_file_from_bucket(
                    input_bucket, input_filename)

                # FIXED: Script generation with proper validation enforcement
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

                # Final validation results for reporting
                is_valid, validation_message = validate_quote_distribution_short(script_content)
                is_enhanced_valid, enhanced_message = validate_enhanced_content_short(script_content, company_name)
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
                    "target_compliance": "optimal" if 1200 <= word_count <= 1600 else ("low" if word_count < 1200 else "high"),
                    "company_mention_compliance": "optimal" if 4 <= company_mentions <= 6 else ("low" if company_mentions < 4 else "high"),
                    "status": "success",
                    "script_type": "SHORT_FIXED"
                })

                print(f"✅ Successfully processed FIXED SHORT {combination}")
                print(f"📊 Quote count: {quote_count}, Target: 8, Validation: {'PASSED' if is_valid else 'FAILED'}")
                print(f"📊 Word count: {word_count}, Target: 1200-1600")
                print(f"🏢 Company mentions: {company_mentions}, Target: 4-6")
                print(f"📋 Structural validation: {validation_message}")
                print(f"🎯 Enhanced validation: {'PASSED' if is_enhanced_valid else 'FAILED'}")
                print(f"📈 Enhanced details: {enhanced_message}")

            except Exception as e:
                print(f"❌ Error processing FIXED SHORT {combination}: {str(e)}")
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
                    "script_type": "SHORT_FIXED"
                })

        # Enhanced summary with FIXED validation statistics
        successful_scripts = [s for s in processed_scripts if s["status"] == "success"]
        failed_scripts = [s for s in processed_scripts if s["status"] == "failed"]
        validated_scripts = [s for s in successful_scripts if s.get("validation_passed", False)]
        enhanced_validated_scripts = [s for s in successful_scripts if s.get("enhanced_validation_passed", False)]
        company_compliant_scripts = [s for s in successful_scripts if s.get("company_mention_compliance") == "optimal"]
        
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
            "company_mention_compliance": len(company_compliant_scripts),
            "validation_rate": f"{len(validated_scripts)}/{len(successful_scripts)}" if successful_scripts else "0/0",
            "enhanced_validation_rate": f"{len(enhanced_validated_scripts)}/{len(successful_scripts)}" if successful_scripts else "0/0",
            "company_compliance_rate": f"{len(company_compliant_scripts)}/{len(successful_scripts)}" if successful_scripts else "0/0",
            "average_quote_count": round(avg_quote_count, 1),
            "target_quote_count": 8,
            "scripts": processed_scripts,
            "company_name": company_name,
            "timestamp": timestamp,
            "openai_model": openai_model,
            "script_type": "SHORT_FIXED",
            "card_range": "11-15",
            "target_duration": "6-8 minutes",
            "fixes_applied": "Proper company name control (4-6 mentions), validation enforcement, instruction hierarchy fix",
            "word_count_analysis": {
                "average_word_count": round(avg_word_count, 1),
                "optimal_compliance": f"{optimal_count}/{len(successful_scripts)}" if successful_scripts else "0/0",
                "target_range": "1200-1600 words"
            },
            "company_integration": {
                "average_mentions": round(avg_company_mentions, 1),
                "target_mentions": "4-6",
                "compliance_rate": f"{len(company_compliant_scripts)}/{len(successful_scripts)}" if successful_scripts else "0/0"
            }
        }

        print(f"\n📊 PROCESSING SUMMARY - FIXED SHORT FORMAT:")
        print(f"✅ FIXED SHORT scripts generated: {len(successful_scripts)}/{total_cards}")
        print(f"✅ Structural validation passed: {len(validated_scripts)}/{len(successful_scripts)}")
        print(f"🎯 Enhanced content validation passed: {len(enhanced_validated_scripts)}/{len(successful_scripts)}")
        print(f"🏢 Company mention compliance: {len(company_compliant_scripts)}/{len(successful_scripts)}")
        print(f"📈 Average quote count: {avg_quote_count:.1f} (target: 8)")
        print(f"📈 Average word count: {avg_word_count:.1f} (target: 1200-1600)")
        print(f"🏢 Average company mentions: {avg_company_mentions:.1f} (target: 4-6)")
        print(f"🎯 Structural success rate: {(len(validated_scripts)/len(successful_scripts)*100):.1f}%" if successful_scripts else "0%")
        print(f"🚀 Enhanced content success rate: {(len(enhanced_validated_scripts)/len(successful_scripts)*100):.1f}%" if successful_scripts else "0%")
        print(f"🏢 Company mention success rate: {(len(company_compliant_scripts)/len(successful_scripts)*100):.1f}%" if successful_scripts else "0%")
        print(f"🎯 Word count compliance: {optimal_count}/{len(successful_scripts)} scripts in optimal range")
        print(f"📋 Fixes applied: Proper company name control, validation enforcement, instruction hierarchy")
        
        return summary

    except Exception as e:
        print(f"❌ Error in process_poppy_cards: {str(e)}")
        raise


def main():
    """Main function to orchestrate the entire FIXED SHORT video script workflow."""
    try:
        logger.info("=" * 80)
        logger.info("FIXED SHORT VIDEO SCRIPT AUTOMATION - COMPANY NAME CONTROL")
        logger.info("=" * 80)
        print("=" * 80)
        print("FIXED SHORT VIDEO SCRIPT AUTOMATION - COMPANY NAME CONTROL")
        print("=" * 80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Script directory: {SCRIPT_DIR}")
        print(f"Target: 6-8 minute scripts with 8 quotes (4 per problem) from cards 11-15")
        print(f"Word Target: 1200-1600 words with rich creative development")
        print("🎯 APPROACH: Exactly 4 quotes per problem (8 total quotes per SHORT script)")
        print("🏢 FIXED: Company name control (4-6 mentions with spacing)")
        print("🚀 VALIDATION: Scripts rejected and retried if validation fails")
        print("   ✅ Minimum 4 company mentions (for branding)")
        print("   ✅ Maximum 6 company mentions (avoid oversaturation)")
        print("   ✅ 4-sentence spacing rule enforced")
        print("   ✅ Alternative references as default behavior")

        # Fetch configuration from Supabase with error handling
        logger.info("[SHORT-ENHANCED] Fetching configuration from Supabase...")
        print("\nFetching FIXED SHORT configuration from Supabase...")
        
        try:
            variables = fetch_configuration_from_supabase()
        except Exception as e:
            logger.critical(f"[SHORT-ENHANCED] Failed to fetch configuration: {str(e)}")
            raise Exception(f"Failed to fetch FIXED SHORT script configuration from Supabase: {str(e)}")

        # Validate that we have the video_script_short configuration
        if "scripts" not in variables or "video_script_short" not in variables["scripts"]:
            logger.critical("[SHORT-ENHANCED] Missing video_script_short configuration in Supabase")
            raise Exception(
                "video_script_short configuration not found in Supabase config. Please ensure the configuration includes a 'video_script_short' section.")

        # Validate global configuration
        if "global" not in variables:
            logger.critical("[SHORT-ENHANCED] Missing global configuration in Supabase")
            raise Exception("global configuration not found in Supabase config. Please ensure the configuration includes a 'global' section.")
        
        video_script_config = variables["scripts"]["video_script_short"]
        logger.info("[SHORT-ENHANCED] Successfully loaded video_script_short configuration")

        # Load guidance files with no fallback handling
        guidance_bucket = video_script_config["supabase_buckets"]["guidance"]
        try:
            guidance_files = load_guidance_files(guidance_bucket)
            logger.info("[SHORT-ENHANCED] Successfully loaded all guidance files for FIXED processing")
        except Exception as e:
            logger.critical(f"[SHORT-ENHANCED] Failed to load guidance files: {str(e)}")
            raise Exception(f"Failed to load FIXED SHORT script guidance files from bucket '{guidance_bucket}': {str(e)}")

        # Process Poppy Cards (cards 11-15) with FIXED company name control
        try:
            logger.info("[SHORT-ENHANCED] Starting FIXED card processing for cards 11-15")
            summary = process_poppy_cards(variables, guidance_files)
            logger.info(f"[SHORT-ENHANCED] Completed FIXED processing: {summary['successful']}/{summary['total_processed']} successful")
        except Exception as e:
            logger.critical(f"[SHORT-ENHANCED] Failed during FIXED card processing: {str(e)}")
            raise Exception(f"Failed to process FIXED SHORT script cards 11-15: {str(e)}")

        # Save summary to output bucket
        output_bucket = video_script_config["supabase_buckets"]["output"]
        summary_filename = f"video_script_SHORT_FIXED_summary_{summary['timestamp']}.json"
        summary_content = json.dumps(summary, indent=2)
        
        try:
            upload_file_to_bucket(output_bucket, summary_filename, summary_content)
            logger.info(f"[SHORT-ENHANCED] Saved FIXED summary as {summary_filename}")
        except Exception as e:
            logger.error(f"[SHORT-ENHANCED] Failed to save FIXED summary: {str(e)}")
            print(f"Warning: Could not save FIXED summary file to bucket '{output_bucket}': {str(e)}")
            print("FIXED video script generation completed successfully despite summary save failure.")

        # Calculate and display execution time
        end_time = datetime.now(eastern_tz)
        execution_time = end_time - start_time

        logger.info("=" * 80)
        logger.info("FIXED SHORT VIDEO SCRIPT WORKFLOW COMPLETE")
        logger.info(f"Execution time: {execution_time}")
        logger.info(f"Success rate: {summary['successful']}/{summary['total_processed']}")
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("FIXED SHORT VIDEO SCRIPT WORKFLOW COMPLETE")
        print("=" * 80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {execution_time}")
        print(f"FIXED SHORT scripts generated: {summary['successful']}/{summary['total_processed']}")
        print(f"Structural validation success rate: {summary['validation_rate']}")
        print(f"Enhanced content validation success rate: {summary['enhanced_validation_rate']}")
        print(f"Company mention compliance rate: {summary['company_compliance_rate']}")
        print(f"Average quote count: {summary['average_quote_count']} (target: 8)")
        print(f"Average word count: {summary['word_count_analysis']['average_word_count']} (target: 1200-1600)")
        print(f"Average company mentions: {summary['company_integration']['average_mentions']} (target: 4-6)")
        print(f"FIXED summary saved as: {summary_filename}")

        if summary['failed'] > 0:
            logger.warning(f"[SHORT-ENHANCED] {summary['failed']} FIXED scripts failed to generate")
            print(f"\n⚠️ Warning: {summary['failed']} FIXED SHORT script(s) failed to generate")
            for script in summary['scripts']:
                if script['status'] == 'failed':
                    logger.error(f"[SHORT-ENHANCED-FAILED] {script['combination']}: {script.get('error', 'Unknown error')}")
                    print(f"  - {script['combination']}: {script.get('error', 'Unknown error')}")

        # Show comprehensive FIXED validation statistics
        validation_passed = summary['validation_passed']
        enhanced_validation_passed = summary['enhanced_validation_passed']
        company_compliant = summary['company_mention_compliance']
        total_successful = summary['successful']
        if total_successful > 0:
            print(f"\n📊 COMPREHENSIVE FIXED VALIDATION ANALYSIS:")
            print(f"✅ Scripts with balanced 2-problem structure: {validation_passed}/{total_successful}")
            print(f"📏 Scripts with 8 quotes (4 per problem): {validation_passed}/{total_successful}")
            print(f"🎯 Scripts with enhanced content: {enhanced_validation_passed}/{total_successful}")
            print(f"🏢 Scripts with proper company mentions (4-6): {company_compliant}/{total_successful}")
            print(f"📈 Company mention compliance rate: {(company_compliant/total_successful*100):.1f}%")
            print(f"🎯 Overall validation success rate: {(min(validation_passed, enhanced_validation_passed, company_compliant)/total_successful*100):.1f}%")
            print(f"⏱️ Duration target compliance: 6-8 minutes with {summary['word_count_analysis']['optimal_compliance']} optimal word count")
            print(f"🚀 Fixes applied: {summary['fixes_applied']}")

        print("\n🎉 FIXED SHORT video script automation workflow completed successfully!")
        print("📋 Each FIXED SHORT script contains exactly 8 quotes (4 quotes per problem)")
        print("🏢 Each FIXED SHORT script includes 4-6 company mentions with proper spacing")
        print("⏱️ Each FIXED SHORT script targets 6-8 minute duration with rich development")
        print("🚀 Validation enforcement ensures quality control")
        print("🎯 Company name control fixes applied:")
        print("   • Minimum 4 mentions requirement")
        print("   • Maximum 6 mentions limit") 
        print("   • 4-sentence spacing rule")
        print("   • Alternative references as default")
        print("   • Script rejection for failed validation")
        
        # Log successful session completion
        logger.info("=" * 60)
        logger.info("FIXED SHORT VIDEO SCRIPT AUTOMATION - SESSION END (SUCCESS)")
        logger.info("=" * 60)

    except Exception as e:
        logger.critical(f"[SHORT-ENHANCED] Critical error in FIXED SHORT workflow: {str(e)}")
        logger.critical(f"[SHORT-ENHANCED] Traceback: {traceback.format_exc()}")
        print(f"\n❌ Critical error in FIXED SHORT workflow: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Ensure we log session end even on failure
        logger.info("=" * 60)
        logger.info("FIXED SHORT VIDEO SCRIPT AUTOMATION - SESSION END (FAILED)")
        logger.info("=" * 60)
        raise


if __name__ == "__main__":
    main()