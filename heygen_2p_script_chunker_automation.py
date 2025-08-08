#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HeyGen 2-Problem Script Chunker Automation Workflow

This script processes existing 2-problem video scripts and intelligently chunks them for HeyGen templates:
- Reads scripts from two-problem-script-drafts bucket only
- Applies smart content-aware splitting (respects sentence boundaries, preserves quotes)
- Chunks scripts according to 2-problem slide structure (10 slides)
- Comments out non-audio segments for video generator script
- Saves chunked scripts with HeyGen placeholder mappings back to same Supabase bucket

Production workflow that integrates with the ClearlyAgentic configuration management system.
Handles ONLY 2-problem scripts - 4-problem scripts are managed by a separate workflow.
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

# Handle missing environment variables
if not supabase_url or not supabase_service_key:
    print("Error: Missing Supabase credentials")
    exit(1)

supabase: Client = create_client(supabase_url, supabase_service_key)

# Record start time for execution tracking (Eastern Time)
eastern_tz = ZoneInfo("America/New_York")
start_time = datetime.now(eastern_tz)

# 2-Problem Template Configuration (10 slides)
TEMPLATE_2PROB_CONFIG = {
    "template_id": "2637a97f32694dc1a2b672dd6d8e7b22",
    "slide_count": 10,
    "bucket_name": "two-problem-script-drafts",
    "script_type": "2prob",
    "structure": {
        "intro": {"slides": [1], "count": 1},
        "problem_1": {"slides": [2, 3, 4], "count": 3},
        "problem_2": {"slides": [5, 6, 7], "count": 3},
        "outro_cta": {"slides": [8, 9, 10], "count": 3}
    }
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [HEYGEN-2PROB] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"heygen_2prob_chunker_log_{datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}.log")
    ])
logger = logging.getLogger(__name__)

# Log initialization for HeyGen 2-problem workflow
logger.info("=" * 60)
logger.info("HEYGEN 2-PROBLEM SCRIPT CHUNKER AUTOMATION - SESSION START")
logger.info(f"Target: Chunk 2-problem scripts for HeyGen slide templates (10 slides)")
logger.info(f"Processing: Input/Output bucket: two-problem-script-drafts")
logger.info(f"Session ID: {datetime.now(eastern_tz).strftime('%Y%m%d_%H%M%S')}")
logger.info("=" * 60)

# Suppress excessive HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Setup directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_PATH = os.path.join(script_dir, "STATIC_VARS_MAR2025.env")

# =====================================================================
# COMMON FUNCTIONS
# =====================================================================

def fetch_configuration_from_supabase(config_name: Optional[str] = None, config_id: Optional[int] = None) -> Dict:
    """
    Fetch configuration variables from Supabase workflow_configs table

    Args:
        config_name: Name of the configuration to fetch
        config_id: ID of the configuration to fetch

    Returns:
        Configuration variables dictionary
    """
    try:
        logger.info(f"[HEYGEN-2PROB-CONFIG] Fetching configuration from Supabase...")

        # Query based on either name or ID
        if config_id:
            logger.info(f"[HEYGEN-2PROB-CONFIG] Fetching configuration with ID: {config_id}")
            response = supabase.table("workflow_configs").select("*").eq("id", config_id).execute()
        elif config_name:
            logger.info(f"[HEYGEN-2PROB-CONFIG] Fetching configuration with name: {config_name}")
            response = supabase.table("workflow_configs").select("*").eq("config_name", config_name).execute()
        else:
            # If no specific config requested, get the most recent one
            logger.info("[HEYGEN-2PROB-CONFIG] Fetching most recent configuration")
            response = supabase.table("workflow_configs").select("*").order("created_at", desc=True).limit(1).execute()

        # Check if we got any data
        if not response.data or len(response.data) == 0:
            raise Exception("No configuration found in Supabase")

        # Return the variables from the first matching record
        config_data = response.data[0]
        config_name = config_data.get('config_name', 'unnamed')
        logger.info(f"[HEYGEN-2PROB-CONFIG] Successfully fetched configuration: {config_name}")
        return config_data.get("variables", {})

    except Exception as e:
        logger.error(f"[HEYGEN-2PROB-CONFIG] Error fetching configuration from Supabase: {str(e)}")
        raise


def get_template_config(heygen_config: Dict) -> Dict:
    """
    Get 2-problem template configuration from loaded config
    
    Args:
        heygen_config: HeyGen configuration from Supabase
        
    Returns:
        Template configuration dictionary for 2-problem scripts
    """
    # Override defaults with configuration values if provided
    config = TEMPLATE_2PROB_CONFIG.copy()
    config["template_id"] = heygen_config.get("template_2prob_id", config["template_id"])
    config["slide_count"] = heygen_config.get("slide_count_2prob", config["slide_count"])
    config["bucket_name"] = heygen_config.get("bucket_name_2prob", config["bucket_name"])
    
    return config


def download_file_from_bucket(bucket_name: str, file_name: str) -> str:
    """
    Download a file from Supabase storage bucket
    
    Args:
        bucket_name: Name of the Supabase bucket
        file_name: Name of the file to download
        
    Returns:
        File content as string
    """
    try:
        logger.info(f"[HEYGEN-2PROB-DOWNLOAD] Downloading {file_name} from {bucket_name} bucket...")
        response = supabase.storage.from_(bucket_name).download(file_name)

        if response:
            content = response.decode('utf-8')
            logger.info(f"[HEYGEN-2PROB-DOWNLOAD] Successfully downloaded {file_name} ({len(content)} characters)")
            return content
        else:
            raise Exception(f"Failed to download {file_name}")

    except Exception as e:
        logger.error(f"[HEYGEN-2PROB-DOWNLOAD] Error downloading {file_name} from {bucket_name}: {str(e)}")
        raise


def upload_file_to_bucket(bucket_name: str, file_name: str, file_content: str) -> bool:
    """
    Upload a file to Supabase storage bucket
    
    Args:
        bucket_name: Name of the Supabase bucket
        file_name: Name of the file to upload
        file_content: Content to upload
        
    Returns:
        True if successful
    """
    try:
        logger.info(f"[HEYGEN-2PROB-UPLOAD] Uploading {file_name} to {bucket_name} bucket...")

        # Convert string content to bytes
        file_bytes = file_content.encode('utf-8')

        response = supabase.storage.from_(bucket_name).upload(
            file_name, file_bytes, {"content-type": "text/plain"})

        logger.info(f"[HEYGEN-2PROB-UPLOAD] Successfully uploaded {file_name} to {bucket_name}")
        return True

    except Exception as e:
        logger.error(f"[HEYGEN-2PROB-UPLOAD] Error uploading {file_name} to {bucket_name}: {str(e)}")
        raise


def list_files_in_bucket(bucket_name: str) -> List[str]:
    """
    List all files in a Supabase storage bucket
    
    Args:
        bucket_name: Name of the Supabase bucket
        
    Returns:
        List of file names
    """
    try:
        logger.info(f"[HEYGEN-2PROB-LIST] Listing files in {bucket_name} bucket...")
        response = supabase.storage.from_(bucket_name).list()

        if response:
            file_names = [file['name'] for file in response if file['name'].endswith('.txt')]
            logger.info(f"[HEYGEN-2PROB-LIST] Found {len(file_names)} .txt files in {bucket_name}")
            return file_names
        else:
            return []

    except Exception as e:
        logger.error(f"[HEYGEN-2PROB-LIST] Error listing files in {bucket_name}: {str(e)}")
        return []

# =====================================================================
# SCRIPT ANALYSIS FUNCTIONS
# =====================================================================

def parse_script_structure(script_content: str) -> Dict[str, any]:
    """
    Parse 2-problem script into logical sections identifying intro, problems, and outro
    
    Args:
        script_content: Raw script content
        
    Returns:
        Parsed sections with content for 2 problems
    """
    sections = {"intro": "", "problems": [], "outro": ""}
    
    try:
        logger.info(f"[HEYGEN-2PROB-PARSE] Parsing 2-problem script structure ({len(script_content)} characters)")
        
        # Try structured parsing first
        sections = parse_script_by_markers(script_content)
        
        # Check content retention after first attempt
        total_parsed_length = len(sections["intro"]) + sum(len(p) for p in sections["problems"]) + len(sections["outro"])
        content_retention = total_parsed_length / len(script_content) if script_content else 0
        logger.info(f"[HEYGEN-2PROB-PARSE] Structured parsing: {content_retention:.2%} retention ({total_parsed_length}/{len(script_content)} chars)")
        
        # If structured parsing didn't work well, try content-based parsing
        if not sections["problems"] or content_retention < 0.8:
            logger.info("[HEYGEN-2PROB-PARSE] Trying content-based parsing...")
            sections = parse_script_by_content_patterns(script_content)
            
            # Check retention again
            total_parsed_length = len(sections["intro"]) + sum(len(p) for p in sections["problems"]) + len(sections["outro"])
            content_retention = total_parsed_length / len(script_content) if script_content else 0
            logger.info(f"[HEYGEN-2PROB-PARSE] Content-based parsing: {content_retention:.2%} retention")
        
        # If we're still losing too much content, force aggressive split
        if content_retention < 0.8:
            logger.warning(f"[HEYGEN-2PROB-PARSE] Low content retention ({content_retention:.2%}), using aggressive parsing")
            sections = parse_script_aggressively(script_content)
            
            # Final check
            total_parsed_length = len(sections["intro"]) + sum(len(p) for p in sections["problems"]) + len(sections["outro"])
            content_retention = total_parsed_length / len(script_content) if script_content else 0
            logger.info(f"[HEYGEN-2PROB-PARSE] Aggressive parsing: {content_retention:.2%} retention")
        
        logger.info(f"[HEYGEN-2PROB-PARSE] Final: intro={len(sections['intro'])}, problems={len(sections['problems'])} ({[len(p) for p in sections['problems']]}), outro={len(sections['outro'])}")
        return sections
        
    except Exception as e:
        logger.error(f"[HEYGEN-2PROB-PARSE] Error parsing script structure: {str(e)}")
        # Ultimate fallback: use aggressive parsing
        logger.info("[HEYGEN-2PROB-PARSE] Using aggressive parsing as fallback")
        return parse_script_aggressively(script_content)


def parse_script_aggressively(script_content: str) -> Dict[str, any]:
    """
    Aggressive parsing that ensures we capture ALL content with maximum retention for 2 problems
    
    Args:
        script_content: Raw script content
        
    Returns:
        Parsed sections with maximum content retention for exactly 2 problems
    """
    sections = {"intro": "", "problems": [], "outro": ""}
    
    logger.info("[HEYGEN-2PROB-PARSE] Starting aggressive parsing for maximum content retention (2 problems)")
    
    total_length = len(script_content)
    
    # Use a balanced split: 25% intro, 50% problems (2 x 25%), 25% outro
    intro_end = int(total_length * 0.25)
    outro_start = int(total_length * 0.75)
    
    # Adjust for sentence boundaries
    intro_chunk = script_content[:intro_end + 300]
    intro_sentence_end = intro_chunk.rfind('. ')
    if intro_sentence_end > intro_end * 0.7:
        intro_end = intro_sentence_end + 1
    
    outro_chunk = script_content[outro_start - 300:outro_start + 300]
    outro_sentence_start = outro_chunk.find('. ')
    if outro_sentence_start > 200:
        outro_start = outro_start - 300 + outro_sentence_start + 1
    
    # Extract sections
    sections["intro"] = script_content[:intro_end].strip()
    sections["outro"] = script_content[outro_start:].strip()
    
    # Split the middle content into exactly 2 problems
    main_content = script_content[intro_end:outro_start].strip()
    main_length = len(main_content)
    
    if main_length > 0:
        problem_length = main_length // 2
        
        # Problem 1: first half
        problem_1_end = problem_length
        
        # Try to find a good sentence boundary
        if problem_1_end < main_length:
            lookahead_chunk = main_content[problem_1_end:min(problem_1_end + 200, main_length)]
            sentence_end = lookahead_chunk.find('. ')
            if sentence_end != -1:
                problem_1_end = problem_1_end + sentence_end + 1
        
        problem_1_text = main_content[:problem_1_end].strip()
        problem_2_text = main_content[problem_1_end:].strip()
        
        sections["problems"] = [problem_1_text, problem_2_text]
    else:
        # If no main content, split the entire script into 2 parts
        logger.warning("[HEYGEN-2PROB-PARSE] No main content found, splitting entire script into 2 problems")
        chunk_size = total_length // 2
        
        problem_1_text = script_content[:chunk_size].strip()
        problem_2_text = script_content[chunk_size:].strip()
        
        sections["problems"] = [problem_1_text, problem_2_text]
        sections["intro"] = ""
        sections["outro"] = ""
    
    # Ensure we have exactly 2 problems
    while len(sections["problems"]) < 2:
        sections["problems"].append("")
    
    sections["problems"] = sections["problems"][:2]
    
    # Validate we captured all content
    total_parsed = len(sections["intro"]) + sum(len(p) for p in sections["problems"]) + len(sections["outro"])
    retention = total_parsed / total_length if total_length > 0 else 0
    
    logger.info(f"[HEYGEN-2PROB-PARSE] Aggressive results: retention={retention:.2%} ({total_parsed}/{total_length} chars)")
    
    return sections


def parse_script_by_markers(script_content: str) -> Dict[str, any]:
    """
    ALIGNED WITH 4P: Parse script using flexible problem markers (handles any problem numbers)
    """
    sections = {"intro": "", "problems": [], "outro": ""}
    
    lines = script_content.split('\n')
    current_section = "intro"
    current_content = []
    
    for line in lines:
        # FIXED: Check for problem marker with any number - matches actual format **[PROBLEM 1:
        if re.match(r'\*\*\[Problem\s+\d+:', line, re.IGNORECASE):
            # Save previous section
            if current_section == "intro":
                sections["intro"] = '\n'.join(current_content).strip()
            elif current_section == "problem" and current_content:
                sections["problems"].append('\n'.join(current_content).strip())
            
            # Start new problem section
            current_section = "problem"
            current_content = [line]  # Include the problem header
            
        elif any(marker in line.lower() for marker in ['**[outro', 'implementing kluster', 'in conclusion']):
            # Save current problem if any
            if current_section == "problem" and current_content:
                sections["problems"].append('\n'.join(current_content).strip())
            
            # Start outro section
            current_section = "outro"
            current_content = [line]
            
        else:
            current_content.append(line)
    
    # Save final section
    if current_section == "intro" and current_content:
        sections["intro"] = '\n'.join(current_content).strip()
    elif current_section == "problem" and current_content:
        sections["problems"].append('\n'.join(current_content).strip())
    elif current_section == "outro" and current_content:
        sections["outro"] = '\n'.join(current_content).strip()
    
    # Ensure we have exactly 2 problems
    while len(sections["problems"]) < 2:
        sections["problems"].append("")
    
    if len(sections["problems"]) > 2:
        sections["problems"] = sections["problems"][:2]
    
    return sections


def parse_script_by_content_patterns(script_content: str) -> Dict[str, any]:
    """
    ALIGNED WITH 4P: Fallback parsing method using content patterns for 2-problem scripts
    """
    sections = {"intro": "", "problems": [], "outro": ""}
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in script_content.split('\n\n') if p.strip()]
    
    intro_end_idx = 0
    outro_start_idx = len(paragraphs)
    
    # Look for outro indicators (same as 4p)
    for i, paragraph in enumerate(paragraphs):
        if any(indicator in paragraph.lower() for indicator in [
            'to wrap up', 'in conclusion', 'remember', 'until next time', 
            'call-to-action', 'download our', 'linked in the description',
            'implementing kluster'
        ]):
            outro_start_idx = i
            break
    
    # Look for intro end (same as 4p)
    for i, paragraph in enumerate(paragraphs):
        if any(indicator in paragraph.lower() for indicator in [
            'our focus will be', 'these problems', 'first problem', 
            'problem 1', 'challenge 1', 'let\'s dive'
        ]):
            intro_end_idx = i
            break
    
    # Extract sections
    if intro_end_idx > 0:
        sections["intro"] = "\n\n".join(paragraphs[:intro_end_idx + 1])
    
    if outro_start_idx < len(paragraphs):
        sections["outro"] = "\n\n".join(paragraphs[outro_start_idx:])
    
    # Everything in the middle is problems/main content
    main_content_paragraphs = paragraphs[intro_end_idx + 1:outro_start_idx]
    
    if main_content_paragraphs:
        if len(main_content_paragraphs) >= 2:
            # Divide paragraphs into 2 roughly equal groups (adapted from 4p logic)
            chunk_size = len(main_content_paragraphs) // 2
            for i in range(2):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < 1 else len(main_content_paragraphs)
                problem_paragraphs = main_content_paragraphs[start_idx:end_idx]
                if problem_paragraphs:
                    sections["problems"].append("\n\n".join(problem_paragraphs))
        else:
            sections["problems"] = main_content_paragraphs
            while len(sections["problems"]) < 2:
                sections["problems"].append("")
    
    # Ensure we have exactly 2 problems
    while len(sections["problems"]) < 2:
        sections["problems"].append("")
    
    sections["problems"] = sections["problems"][:2]
    
    return sections


def preserve_customer_quotes(text: str) -> List[Tuple[int, int]]:
    """
    Identify and protect customer quote boundaries
    """
    quote_patterns = [
        r'"[^"]*"',  # Standard quotes
        r'"[^"]*"',  # Smart quotes
        r'One user noted[^.]*\.',
        r'Another mentioned[^.]*\.',
        r'A customer shared[^.]*\.',
        r'As one [^.]*said[^.]*\.'
    ]
    
    quotes = []
    for pattern in quote_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            quotes.append((match.start(), match.end()))
    
    # Sort and merge overlapping quotes
    quotes.sort()
    merged_quotes = []
    for start, end in quotes:
        if merged_quotes and start <= merged_quotes[-1][1]:
            merged_quotes[-1] = (merged_quotes[-1][0], max(end, merged_quotes[-1][1]))
        else:
            merged_quotes.append((start, end))
    
    return merged_quotes


def find_optimal_split_point(text: str, target_position: int, quotes: List[Tuple[int, int]]) -> int:
    """
    Find best split point near target position avoiding quote breaks
    """
    # Check if target position is within a quote
    for quote_start, quote_end in quotes:
        if quote_start <= target_position <= quote_end:
            target_position = quote_end + 1
    
    # Look for sentence endings near target position
    search_range = 200
    start_search = max(0, target_position - search_range)
    end_search = min(len(text), target_position + search_range)
    
    # Find sentence endings
    sentence_endings = []
    for match in re.finditer(r'[.!?]\s+', text[start_search:end_search]):
        abs_position = start_search + match.end()
        
        # Check if this split point is within a quote
        in_quote = any(quote_start <= abs_position <= quote_end for quote_start, quote_end in quotes)
        if not in_quote:
            sentence_endings.append(abs_position)
    
    if sentence_endings:
        # Find closest sentence ending to target
        closest = min(sentence_endings, key=lambda x: abs(x - target_position))
        return closest
    
    # Look for paragraph breaks
    for match in re.finditer(r'\n\s*\n', text[start_search:end_search]):
        abs_position = start_search + match.start()
        in_quote = any(quote_start <= abs_position <= quote_end for quote_start, quote_end in quotes)
        if not in_quote:
            return abs_position
    
    return target_position


def split_content_intelligently(text: str, target_char_count: int, max_char_count: int) -> List[str]:
    """
    Split text into chunks respecting boundaries and character limits
    """
    if len(text) <= max_char_count:
        return [text]
    
    quotes = preserve_customer_quotes(text)
    chunks = []
    remaining_text = text
    
    while len(remaining_text) > max_char_count:
        if len(remaining_text) > target_char_count:
            split_point = find_optimal_split_point(remaining_text, target_char_count, quotes)
        else:
            split_point = len(remaining_text)
        
        chunk = remaining_text[:split_point].strip()
        if chunk:
            chunks.append(chunk)
        
        remaining_text = remaining_text[split_point:].strip()
        quotes = [(start - split_point, end - split_point) for start, end in quotes 
                 if end > split_point]
        quotes = [(max(0, start), end) for start, end in quotes if end > 0]
    
    if remaining_text.strip():
        chunks.append(remaining_text.strip())
    
    return chunks

# =====================================================================
# CONTENT ALLOCATION FUNCTIONS
# =====================================================================

def allocate_content_to_slides(parsed_sections: Dict[str, any], template_config: Dict, heygen_config: Dict) -> Dict[int, str]:
    """
    Map content sections to specific slides based on 2-problem template structure
    """
    slide_contents = {}
    structure = template_config["structure"]
    
    # Get character limits from configuration
    max_chars_per_slide = heygen_config.get("max_chars_per_slide", 1500)
    target_chars_per_slide = heygen_config.get("target_chars_per_slide", 1200)
    
    logger.info(f"[HEYGEN-2PROB-ALLOCATE] Allocating content: {len(parsed_sections['intro'])} chars intro, {len(parsed_sections['problems'])} problems, {len(parsed_sections['outro'])} chars outro")
    
    try:
        # Allocate intro content (slide 1)
        intro_slides = structure["intro"]["slides"]
        if parsed_sections["intro"]:
            intro_chunks = split_content_intelligently(
                parsed_sections["intro"], 
                target_chars_per_slide, 
                max_chars_per_slide
            )
            for i, slide_num in enumerate(intro_slides):
                if i < len(intro_chunks):
                    slide_contents[slide_num] = intro_chunks[i]
                else:
                    slide_contents[slide_num] = ""
        
        # Allocate 2 problem content (slides 2-7, 3 slides per problem)
        problems = parsed_sections["problems"]
        problem_keys = ["problem_1", "problem_2"]
        
        logger.info(f"[HEYGEN-2PROB-ALLOCATE] Distributing 2 problems across 2 problem sections (3 slides each)")
        
        for i, problem_key in enumerate(problem_keys):
            if i < len(problems) and problems[i].strip():
                problem_content = problems[i]
                problem_slides = structure[problem_key]["slides"]
                num_slides_for_problem = len(problem_slides)  # Should be 3
                
                logger.info(f"[HEYGEN-2PROB-ALLOCATE] Problem {i+1}: {len(problem_content)} chars -> {num_slides_for_problem} slides {problem_slides}")
                
                if num_slides_for_problem > 1:
                    target_per_slide = len(problem_content) // num_slides_for_problem
                    target_per_slide = max(target_per_slide, 150)
                    
                    problem_chunks = split_content_into_n_chunks(
                        problem_content,
                        num_slides_for_problem,
                        target_per_slide
                    )
                else:
                    problem_chunks = [problem_content]
                
                for j, slide_num in enumerate(problem_slides):
                    if j < len(problem_chunks):
                        slide_contents[slide_num] = problem_chunks[j]
                    else:
                        slide_contents[slide_num] = ""
            else:
                problem_slides = structure[problem_key]["slides"]
                for slide_num in problem_slides:
                    slide_contents[slide_num] = ""
        
        # Allocate outro content (slides 8-10)
        outro_slides = structure["outro_cta"]["slides"]
        if parsed_sections["outro"]:
            num_outro_slides = len(outro_slides)  # Should be 3
            if num_outro_slides > 1:
                target_per_slide = len(parsed_sections["outro"]) // num_outro_slides
                target_per_slide = max(target_per_slide, 150)
                
                outro_chunks = split_content_into_n_chunks(
                    parsed_sections["outro"],
                    num_outro_slides,
                    target_per_slide
                )
            else:
                outro_chunks = [parsed_sections["outro"]]
                
            for i, slide_num in enumerate(outro_slides):
                if i < len(outro_chunks):
                    slide_contents[slide_num] = outro_chunks[i]
                else:
                    slide_contents[slide_num] = ""
        
        # Ensure all 10 slides have content
        for slide_num in range(1, template_config["slide_count"] + 1):
            if slide_num not in slide_contents:
                slide_contents[slide_num] = ""
        
        non_empty_slides = sum(1 for content in slide_contents.values() if content.strip())
        logger.info(f"[HEYGEN-2PROB-ALLOCATE] Final allocation: {non_empty_slides} non-empty slides out of {template_config['slide_count']}")
        
        return slide_contents
        
    except Exception as e:
        logger.error(f"[HEYGEN-2PROB-ALLOCATE] Error allocating content to slides: {str(e)}")
        return allocate_content_evenly(parsed_sections, template_config, heygen_config)


def split_content_into_n_chunks(text: str, n_chunks: int, target_chars_per_chunk: int) -> List[str]:
    """
    Force split content into exactly n chunks
    """
    if n_chunks <= 1:
        return [text]
    
    if len(text) < n_chunks * 50:
        chunk_size = len(text) // n_chunks
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < n_chunks - 1 else len(text)
            chunk = text[start:end].strip()
            chunks.append(chunk if chunk else "")
        return chunks
    
    quotes = preserve_customer_quotes(text)
    chunks = []
    remaining_text = text
    target_size = len(text) // n_chunks
    
    for i in range(n_chunks):
        if i == n_chunks - 1:
            chunks.append(remaining_text.strip())
            break
        
        if len(remaining_text) > target_size:
            split_point = find_optimal_split_point(remaining_text, target_size, quotes)
        else:
            split_point = len(remaining_text)
        
        chunk = remaining_text[:split_point].strip()
        chunks.append(chunk if chunk else "")
        
        remaining_text = remaining_text[split_point:].strip()
        quotes = [(start - split_point, end - split_point) for start, end in quotes 
                 if end > split_point]
        quotes = [(max(0, start), end) for start, end in quotes if end > 0]
    
    while len(chunks) < n_chunks:
        chunks.append("")
    
    return chunks[:n_chunks]


def allocate_content_evenly(parsed_sections: Dict[str, any], template_config: Dict, heygen_config: Dict) -> Dict[int, str]:
    """
    Fallback allocation method - distribute content evenly across 10 slides
    """
    logger.info("[HEYGEN-2PROB-ALLOCATE] Starting even content allocation for 10 slides")
    
    # Get character limits from configuration
    max_chars_per_slide = heygen_config.get("max_chars_per_slide", 1500)
    target_chars_per_slide = heygen_config.get("target_chars_per_slide", 1200)
    
    # Combine all content
    all_content = []
    
    if parsed_sections["intro"] and parsed_sections["intro"].strip():
        all_content.append(parsed_sections["intro"])
    
    for i, problem in enumerate(parsed_sections["problems"]):
        if problem and problem.strip():
            all_content.append(problem)
    
    if parsed_sections["outro"] and parsed_sections["outro"].strip():
        all_content.append(parsed_sections["outro"])
    
    combined_content = "\n\n".join(all_content)
    logger.info(f"[HEYGEN-2PROB-ALLOCATE] Combined content length: {len(combined_content)} chars from {len(all_content)} sections")
    
    # Split evenly across all 10 slides
    total_slides = template_config["slide_count"]  # 10
    
    if len(combined_content) > 0:
        target_chunk_size = max(200, len(combined_content) // total_slides)
        max_chunk_size = min(max_chars_per_slide, len(combined_content) // 3)
    else:
        target_chunk_size = 200
        max_chunk_size = max_chars_per_slide
    
    chunks = split_content_intelligently(
        combined_content,
        target_chunk_size,
        max_chunk_size
    )
    
    slide_contents = {}
    
    for i in range(total_slides):
        if i < len(chunks):
            slide_contents[i + 1] = chunks[i]
        else:
            slide_contents[i + 1] = ""
    
    return slide_contents


def format_chunked_output(slide_contents: Dict[int, str], original_content: str) -> str:
    """
    Format chunked content for HeyGen API with 2prob placeholder mappings
    """
    output_lines = []
    
    # Add header
    output_lines.extend([
        "# HeyGen Chunked 2-Problem Script",
        f"# Template Type: 2prob",
        f"# Generated: {datetime.now(eastern_tz).strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Total Slides: {len(slide_contents)}",
        ""
    ])
    
    # Add placeholder mappings for 2prob
    for slide_num in sorted(slide_contents.keys()):
        placeholder = f"{{{{two_prob_slide_{slide_num:02d}_content}}}}"
        content = slide_contents[slide_num].strip()
        char_count = len(content)
        
        output_lines.extend([
            f"# Slide {slide_num} ({char_count} characters)",
            f"{placeholder}",
            content,
            ""
        ])
    
    # Add summary statistics
    total_chars = sum(len(content) for content in slide_contents.values())
    avg_chars = total_chars / len(slide_contents) if slide_contents else 0
    non_empty_slides = sum(1 for content in slide_contents.values() if content.strip())
    
    output_lines.extend([
        "# Chunking Summary",
        f"# Total characters: {total_chars}",
        f"# Average per slide: {avg_chars:.1f}",
        f"# Non-empty slides: {non_empty_slides}",
        f"# Original length: {len(original_content)}"
    ])
    
    return "\n".join(output_lines)

# =====================================================================
# COMMENT OUT NON-AUDIO SEGMENTS FUNCTION
# =====================================================================

def comment_out_non_audio_segments(chunked_content: str) -> str:
    """
    SURGICAL commenting: Only comment out specific non-audio segments.
    INPUT: Clean content with NO existing # prefixes
    OUTPUT: Mostly clean content with ONLY targeted lines commented out
    
    This function identifies and comments out:
    1. Section markers like **[Intro]**, **[Outro]**, **[Transition]**  
    2. Final synopsis section (only after the LAST '---' separator)
    
    Args:
        chunked_content (str): The chunked script content
        
    Returns:
        str: Content with non-audio segments commented out
    """
    
    lines = chunked_content.split('\n')
    processed_lines = []
    
    # FIND THE FINAL '---' - only content after THIS should be synopsis
    last_separator_index = -1
    for i, line in enumerate(lines):
        if line.strip() == '---':
            last_separator_index = i
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Always keep separators and empty lines as-is
        if not line_stripped or line_stripped == '---':
            processed_lines.append(line)
            continue
        
        # Only comment content after the FINAL '---' AND skip chunking summary lines
        in_final_synopsis = (last_separator_index != -1 and i > last_separator_index)
        if in_final_synopsis:
            # Don't double-comment lines that are already chunking metadata
            if line_stripped.startswith('# '):
                processed_lines.append(line)  # Keep existing metadata comments
            else:
                processed_lines.append(f"# {line}")  # Comment out synopsis prose
            continue
        
        # Comment out ONLY specific section markers (exact matches only)
        markers_to_comment = [
            '**Intro (', '**INTRO (', 
            '**Outro (', '**OUTRO (',
            '**Transition (', '**TRANSITION (',
            '**[Intro]**', '**[INTRO]**', '[INTRO]', '[intro]',
            '**[Outro]**', '**[OUTRO]**', '[OUTRO]', '[outro]',
            '**[Transition]**', '**[TRANSITION]**', '[TRANSITION]', '[transition]'
        ]
        
        should_comment = any(marker in line_stripped for marker in markers_to_comment)
        
        if should_comment:
            processed_lines.append(f"# {line}")  # Comment out section markers
        else:
            processed_lines.append(line)  # ✅ KEEP ALL OTHER CONTENT CLEAN FOR SPEAKING
    
    return '\n'.join(processed_lines)

# =====================================================================
# MAIN WORKFLOW FUNCTIONS
# =====================================================================

def process_single_script(filename: str, company_name: str, template_config: Dict, heygen_config: Dict) -> Dict:
    """
    Process one 2-problem script file through complete chunking pipeline
    """
    try:
        logger.info(f"[HEYGEN-2PROB-PROCESS] Processing {filename}")
        
        bucket_name = template_config["bucket_name"]  # two-problem-script-drafts
        
        # Download script content
        script_content = download_file_from_bucket(bucket_name, filename)
        
        # Parse script structure for 2 problems
        parsed_sections = parse_script_structure(script_content)
        
        # Allocate content to 10 slides
        slide_contents = allocate_content_to_slides(parsed_sections, template_config, heygen_config)
        
        # Format output for 2prob template
        formatted_output = format_chunked_output(slide_contents, script_content)
        
        # Comment out non-audio segments
        commented_output = comment_out_non_audio_segments(formatted_output)
        
        # Generate output filename
        chunked_filename = generate_chunked_filename(filename)
        
        # Upload commented chunked script to same bucket
        upload_file_to_bucket(bucket_name, chunked_filename, commented_output)
        
        # Calculate metrics
        total_chars = sum(len(content) for content in slide_contents.values() if content)
        avg_chars = total_chars / len(slide_contents) if slide_contents else 0
        non_empty_slides = sum(1 for content in slide_contents.values() if content.strip())
        
        return {
            "filename": filename,
            "chunked_filename": chunked_filename,
            "script_type": "2prob",
            "slide_count": len(slide_contents),
            "non_empty_slides": non_empty_slides,
            "total_chars": total_chars,
            "avg_chars_per_slide": avg_chars,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"[HEYGEN-2PROB-PROCESS] Error processing {filename}: {str(e)}")
        return {
            "filename": filename,
            "error": str(e),
            "status": "failed"
        }


def generate_chunked_filename(original_filename: str) -> str:
    """
    Generate filename for chunked 2-problem script
    
    Args:
        original_filename: Original script filename
        
    Returns:
        New filename for chunked script
        
    Example:
        kluster_SHORT_FINAL_script_P4_P1_20250720_1310.txt
        -> kluster_SHORT_creative_enhanced_chunked_2prob_script_P4_P1_20250720_1310.txt
    """
    base_name = original_filename.replace('.txt', '')
    
    # Insert chunked indicator for 2-problem scripts
    chunked_name = base_name.replace('script_', 'chunked_2prob_script_')
    
    return chunked_name + '.txt'


def process_all_scripts_for_company(company_name: str, template_config: Dict, heygen_config: Dict) -> Dict:
    """
    Process all 2-problem scripts for a company from two-problem-script-drafts bucket
    """
    logger.info(f"[HEYGEN-2PROB-MAIN] Processing all 2-problem scripts for company: {company_name}")
    
    results = {
        "company_name": company_name,
        "processed_scripts": [],
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "bucket_name": template_config["bucket_name"],
        "timestamp": datetime.now(eastern_tz).strftime("%Y%m%d_%H%M%S")
    }
    
    bucket_name = template_config["bucket_name"]  # two-problem-script-drafts
    logger.info(f"[HEYGEN-2PROB-MAIN] Processing scripts from {bucket_name} bucket...")
    
    # List all files in bucket
    all_files = list_files_in_bucket(bucket_name)
    
    # Filter files for this company (exclude already chunked files)
    # Look for files with the specific pattern: <company>_SHORT_FINAL_script_
    company_files = [f for f in all_files 
                    if f.startswith(company_name) and 'SHORT_FINAL_script_' in f and 'chunked' not in f.lower()]
    
    logger.info(f"[HEYGEN-2PROB-MAIN] Found {len(company_files)} script files for {company_name}")
    
    # Process each file
    for filename in company_files:
        result = process_single_script(filename, company_name, template_config, heygen_config)
        results["processed_scripts"].append(result)
        results["total_processed"] += 1
        
        if result["status"] == "success":
            results["successful"] += 1
        else:
            results["failed"] += 1
    
    logger.info(f"[HEYGEN-2PROB-MAIN] Processing complete: {results['successful']}/{results['total_processed']} scripts chunked successfully")
    return results


def main():
    """Main function to orchestrate the entire HeyGen 2-problem chunking workflow."""
    try:
        print("=" * 80)
        print("HEYGEN 2-PROBLEM SCRIPT CHUNKER AUTOMATION WORKFLOW")
        print("=" * 80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Script directory: {SCRIPT_DIR}")
        print(f"Target: 2-problem scripts → 10 slides")
        print(f"Bucket: two-problem-script-drafts (input/output)")

        # Fetch configuration from Supabase
        logger.info("[HEYGEN-2PROB-MAIN] Fetching configuration from Supabase...")
        print("\nFetching configuration from Supabase...")
        variables = fetch_configuration_from_supabase()

        # Validate that we have the heygen_2prob_chunker configuration
        if "scripts" not in variables or "heygen_2prob_chunker" not in variables["scripts"]:
            raise Exception(
                "heygen_2prob_chunker configuration not found in Supabase config. Please ensure the configuration includes a 'heygen_2prob_chunker' section.")

        if "global" not in variables:
            raise Exception(
                "global configuration not found in Supabase config. Please ensure the configuration includes a 'global' section.")

        heygen_config = variables["scripts"]["heygen_2prob_chunker"]
        global_config = variables["global"]
        company_name = global_config.get("COMPANY_NAME")

        if not company_name:
            raise Exception(
                "COMPANY_NAME not found in global configuration. Please ensure the configuration includes a valid COMPANY_NAME.")

        logger.info(f"[HEYGEN-2PROB-MAIN] Processing 2-problem scripts for company: {company_name}")
        print(f"Processing 2-problem scripts for company: {company_name}")

        # Get template configuration for 2-problem scripts
        template_config = get_template_config(heygen_config)

        # Process all 2-problem scripts for the company
        summary = process_all_scripts_for_company(company_name, template_config, heygen_config)

        # Save summary to same bucket
        bucket_name = template_config["bucket_name"]
        summary_filename = f"heygen_2prob_chunking_summary_{summary['timestamp']}.json"
        summary_content = json.dumps(summary, indent=2)
        upload_file_to_bucket(bucket_name, summary_filename, summary_content)

        # Calculate and display execution time
        end_time = datetime.now(eastern_tz)
        execution_time = end_time - start_time

        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETE")
        print("=" * 80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {execution_time}")
        print(f"2-problem scripts chunked: {summary['successful']}/{summary['total_processed']}")
        print(f"Summary saved as: {summary_filename}")

        if summary['failed'] > 0:
            print(f"\nWarning: {summary['failed']} script(s) failed to chunk")
            for script in summary['processed_scripts']:
                if script['status'] == 'failed':
                    print(f"  - {script['filename']}: {script.get('error', 'Unknown error')}")

        print("\nHeyGen 2-problem script chunking workflow completed successfully!")

    except Exception as e:
        logger.error(f"[HEYGEN-2PROB-MAIN] Critical error in main workflow: {str(e)}")
        print(f"\nCritical error in main workflow: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
