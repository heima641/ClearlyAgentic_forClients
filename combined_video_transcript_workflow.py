#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined YouTube Transcript Processing Workflow Script - - - REVISED FOR VIDEO TRANSCRIPT WORKFLOW

This script combines four sequential processes into a single workflow:

1. MODULE 1: YouTube Transcript Scraping
   - Retrieves a specified number of video transcripts from a chosen YouTube channel or handle
   - Fetches their titles and transcript text
   - Saves the results in a JSON file

2. MODULE 2: Content Chunking for Embeddings
   - Processes the retrieved transcript content by chunking it into smaller pieces suitable for embeddings
   - Enriches the chunks with metadata
   - Saves the results in a second JSON file

3. MODULE 3: Advanced Chunk Processing
   - Further processes the chunked transcript content with additional metadata
   - Prepares the data for embedding generation

4. MODULE 4: Embedding Generation and Pinecone Upsert
   - Generates embeddings for each transcript chunk
   - Upserts the embeddings and metadata to Pinecone vector database
   - Saves a complete record of all processed transcript data

All output files are saved to a Supabase Bucket instead of the AGENTIC_OUTPUT folder.
"""

# =====================================================================
# COMMON IMPORTS AND SETUP
# =====================================================================

import os
import json
import requests
import re
import time
import traceback
import logging
import nltk
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client, Client
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random

# Import OpenAI and Pinecone for Module 4
import openai
from pinecone import Pinecone

# SUP BUCKET OUTPUT SUP BUCKET OUTPUT
# Supabase imports already included above

# =====================================================================
# CONSTANTS AND CONFIGURATION
# =====================================================================

# Path to environment variables file
ENV_FILE_PATH = "STATIC_VARS_MAR2025.env"

# SUP BUCKET OUTPUT SUP BUCKET OUTPUT
# Load environment variables (for any additional env vars needed)
load_dotenv(dotenv_path=ENV_FILE_PATH)

# # SUPABASE BUCKET: initialize Supabase client
supabase_url = os.getenv("VITE_SUPABASE_URL")
supabase_service_key = os.getenv("VITE_SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(supabase_url, supabase_service_key)
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # seconds

# Record start time for execution tracking
start_time = datetime.now()

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')

# Flag to track if we've already shown the punkt_tab error message
_shown_punkt_tab_error = False

# Helper function to avoid repetitive error messages

def log_punkt_tab_error(function_name):
    """Log NLTK punkt tokenizer errors and provide guidance."""
    global _shown_punkt_tab_error
    if not _shown_punkt_tab_error:
        print(f"NLTK Error in {function_name}: punkt tokenizer not found.")
        print("Using fallback methods for text processing.")
        _shown_punkt_tab_error = True

# Configure logging for Module 4
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"youtube_transcript_embedding_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Suppress excessive HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Setup directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_PATH = os.path.join(script_dir, "STATIC_VARS_MAR2025.env")

# Note: No longer using local output directory as files will be saved to Supabase bucket

# =====================================================================
# COMMON FUNCTIONS
# =====================================================================

def fetch_configuration_from_supabase(supabase_url, supabase_service_key, config_name=None, config_id=None):
    """
    Fetch configuration variables from Supabase workflow_configs table

    Args:
        supabase_url (str): Supabase project URL
        supabase_service_key (str): Supabase service role key
        config_name (str, optional): Name of the configuration to fetch
        config_id (int, optional): ID of the configuration to fetch

    Returns:
        dict: Configuration variables
    """
    try:
        # Initialize Supabase client
        supabase: Client = create_client(supabase_url, supabase_service_key)

        # Query based on either name or ID
        if config_id:
            print(f"Fetching configuration with ID: {config_id}")
            response = supabase.table("workflow_configs").select("*").eq("id", config_id).execute()
        elif config_name:
            print(f"Fetching configuration with name: {config_name}")
            response = supabase.table("workflow_configs").select("*").eq("config_name", config_name).execute()
        else:
            # If no specific config requested, get the most recent one
            print("Fetching most recent configuration")
            response = supabase.table("workflow_configs").select("*").order("created_at", desc=True).limit(1).execute()

        # Check if we got any data
        if not response.data or len(response.data) == 0:
            print("No configuration found in Supabase")
            raise Exception("No configuration found in Supabase")

        # Return the variables from the first matching record
        config_data = response.data[0]
        print(f"Successfully fetched configuration: {config_data.get('config_name', 'unnamed')}")
        return config_data.get("variables", {})

    except Exception as e:
        print(f"Error fetching configuration from Supabase: {str(e)}")
        raise

# =====================================================================
# MODULE 1: YOUTUBE TRANSCRIPT SCRAPING IMPLEMENTATION
# =====================================================================

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import time

def get_channel_id(company_name_or_handle, YOUTUBE_API_KEY):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    search_response = youtube.search().list(
        q=company_name_or_handle,
        part="id,snippet",
        type="channel",
        maxResults=5
    ).execute()
    if not search_response["items"]:
        raise ValueError(f"No channels found for: {company_name_or_handle}")
    # Automatically pick the first channel's ID (adjust if needed)
    return search_response["items"][0]["id"]["channelId"]

def extract_video_ids(channel_id, YOUTUBE_API_KEY, num_videos):
    youtube_client = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    channel_response = youtube_client.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()
    uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    video_ids = []
    next_page_token = None
    while len(video_ids) < num_videos:
        playlist_response = youtube_client.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()
        for item in playlist_response["items"]:
            video_ids.append(item["contentDetails"]["videoId"])
            if len(video_ids) >= num_videos:
                break
        next_page_token = playlist_response.get("nextPageToken")
        if not next_page_token:
            break
    return video_ids

def scrape_video_transcript(video_id, YOUTUBE_API_KEY):
    try:
        transcript_entries = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([entry["text"] for entry in transcript_entries])
    except:
        return None
    youtube_client = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    video_response = youtube_client.videos().list(
        part="snippet",
        id=video_id
    ).execute()
    if video_response["items"]:
        title = video_response["items"][0]["snippet"]["title"]
    else:
        title = "No Title"
    return {
        "title": title,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "content": transcript_text
    }

def run_module_1_youtube_transcript_scraping(variables, youtube_json_path):
    """
    Module 1: YouTube Transcript Scraping (YouTube API version)
    Fetches recent video transcripts from a YouTube channel and saves them to a JSON file.
    """
    YOUTUBE_API_KEY = variables["YOUTUBE_API_KEY"]
    COMPANY_NAME_OR_HANDLE = variables["COMPANY_NAME_OR_HANDLE"]
    COMPANY_NAME = variables["COMPANY_NAME"]
    NUM_VIDEOS = int(variables["NUM_VIDEOS"])

    print(f"Finding channel ID for: {COMPANY_NAME_OR_HANDLE}")
    channel_id = get_channel_id(COMPANY_NAME_OR_HANDLE, YOUTUBE_API_KEY)

    print(f"Extracting up to {NUM_VIDEOS} video IDs for channel ID: {channel_id}")
    recent_videos = extract_video_ids(channel_id, YOUTUBE_API_KEY, NUM_VIDEOS)

    print(f"🔹 {COMPANY_NAME} - Latest {NUM_VIDEOS} YouTube Videos:")
    for vid in recent_videos:
        print(vid)

    print("\nPausing for 7 seconds before scraping video transcripts...")
    time.sleep(7)

    scraped_data = []
    for vid in recent_videos:
        print(f"Scraping transcript for video: {vid}")
        data = scrape_video_transcript(vid, YOUTUBE_API_KEY)
        if data:
            scraped_data.append(data)
        time.sleep(2)

    if scraped_data:
        print("\nContent from the first scraped video:")
        print(scraped_data[0])
    else:
        print("\nNo transcripts were available for the videos processed.")

    print("\nPausing for 7 seconds before saving data to JSON file...")
    time.sleep(7)

    with open(youtube_json_path, "w") as f:
        json.dump(scraped_data, f, indent=4)

    print(f"YouTube transcript data saved to {youtube_json_path}")
    return youtube_json_path

def run_module_1_youtube_transcript_scraping(variables, youtube_json_path):
    """Module 1: YouTube Transcript Scraping
    Retrieves a specified number of articles from a chosen content category,
    fetches their titles and text, and saves the results in a JSON file.
    
    Args:
        variables: Configuration variables from Supabase
        youtube_json_path: Path where the YouTube transcript content will be saved
    
    Returns:
        str: Path to the generated JSON file
    """
    print("\n" + "=" * 80)
    print("MODULE 1: YOUTUBE TRANSCRIPT SCRAPING")
    print("=" * 80)
    
    module_start_time = datetime.now()
    
    # Environment variables are already loaded in the global section
    print(f"Using configuration variables passed from main function")
    
    # Use the variables passed from the main function
    try:
        
        # Extract transcript-specific configuration
        if "scripts" not in variables or "transcripts" not in variables["scripts"]:
            raise Exception("No 'scripts.transcripts' section found in Supabase config")
        
        transcript_config = variables["scripts"]["transcripts"]
        
        # Get COMPANY_NAME from global section (since it's not in transcripts section)
        if "global" in variables and "COMPANY_NAME" in variables["global"]:
            company_name = variables["global"]["COMPANY_NAME"]
            print(f"Successfully fetched COMPANY_NAME from Supabase global section: {company_name}")
        else:
            raise Exception("COMPANY_NAME not found in Supabase config")

        # Get SITEMAP_URL from global section (since it's not in transcripts section)
        if "global" in variables and "SITEMAP_URL" in variables["global"]:
            sitemap_url = variables["global"]["SITEMAP_URL"]
            print(f"Successfully fetched SITEMAP_URL from Supabase global section: {sitemap_url}")
        else:
            raise Exception("SITEMAP_URL not found in Supabase config")

        # Get NUM_VIDEOS
        if "NUM_VIDEOS" not in transcript_config:
            raise Exception("NUM_VIDEOS not found in Supabase config")
        num_videos = int(transcript_config["NUM_VIDEOS"])
        print(f"Successfully fetched NUM_VIDEOS from Supabase: {num_videos}")

        # Get COMPANY_NAME_OR_HANDLE
        if "COMPANY_NAME_OR_HANDLE" not in transcript_config:
            raise Exception("COMPANY_NAME_OR_HANDLE not found in Supabase config")
        company_name_or_handle = transcript_config["COMPANY_NAME_OR_HANDLE"]
        print(f"Successfully fetched COMPANY_NAME_OR_HANDLE from Supabase: {company_name_or_handle}")

        # Get YOUTUBE_CATEG_OPTIONS_SITEMAP_URL (optional - can be empty string)
        if "YOUTUBE_CATEG_OPTIONS_SITEMAP_URL" in transcript_config:
            youtube_categ_options_sitemap_url = transcript_config["YOUTUBE_CATEG_OPTIONS_SITEMAP_URL"]
            print(f"Successfully fetched YOUTUBE_CATEG_OPTIONS_SITEMAP_URL from Supabase: {youtube_categ_options_sitemap_url}")
        else:
            youtube_categ_options_sitemap_url = ""
            print("YOUTUBE_CATEG_OPTIONS_SITEMAP_URL not found in Supabase, using empty string")



    except Exception as e:
        print(f"ERROR: Failed to fetch required configuration from Supabase: {str(e)}")
        print("Script execution will be halted as required variables are missing.")
        return None


    print(f"Using configuration: COMPANY_NAME={company_name}, NUM_VIDEOS={num_videos}, COMPANY_NAME_OR_HANDLE={company_name_or_handle}")
    print(f"YOUTUBE_CATEG_OPTIONS_SITEMAP_URL: {youtube_categ_options_sitemap_url}")
    
    # --- Part 1: Generate a list of the YouTube video IDs ---
    if "YOUTUBE_API_KEY" not in variables or not variables["YOUTUBE_API_KEY"]:
        # Environment variables already loaded in global section
        variables["YOUTUBE_API_KEY"] = os.environ.get("YOUTUBE_API_KEY")
    YOUTUBE_API_KEY = variables["YOUTUBE_API_KEY"]
    channel_id = get_channel_id(company_name_or_handle, YOUTUBE_API_KEY)
    recent_videos = extract_video_ids(channel_id, YOUTUBE_API_KEY, num_videos)

    print(f"\n🔹 {company_name} - Latest {num_videos} YouTube Videos for channel/handle '{company_name_or_handle}':")
    for vid in recent_videos:
        print(f"https://www.youtube.com/watch?v={vid}")

    print("\nPausing for 7 seconds before scraping video transcripts...")
    time.sleep(7)

    scraped_data = []
    for vid in recent_videos:
        print(f"Scraping transcript for video: {vid}")
        data = scrape_video_transcript(vid, YOUTUBE_API_KEY)
        if data:
            scraped_data.append(data)
        time.sleep(2)

    if scraped_data:
        print("\nContent from the first scraped video:")
        print(scraped_data[0])
    else:
        print("\nNo content was scraped.")
        return None

    # 7-second pause between scraping and saving data
    print("\nPausing for 7 seconds before saving data to JSON file...")
    time.sleep(7)

    # --- Part 3: Save the scraped data to a JSON file and upload to Supabase bucket ---
    with open(youtube_json_path, "w") as f:
        json.dump(scraped_data, f, indent=4)
    
    # Upload to Supabase bucket
    with open(youtube_json_path, "rb") as f:
        data = f.read()
    res = supabase.storage.from_("agentic-output").upload(os.path.basename(youtube_json_path), data)
    if not res:
        raise Exception(f"Failed to upload {youtube_json_path} to Supabase bucket")
    print(f"Uploaded {os.path.basename(youtube_json_path)} to Supabase bucket")
    
    print(f"Transcript data saved to {youtube_json_path}")
    print(f"Module 1 execution completed in {datetime.now() - module_start_time}")
    
    return youtube_json_path

# =====================================================================
# MODULE 2: CONTENT CHUNKING IMPLEMENTATION
# =====================================================================

# --- Analyzer Class for Sentiment Analysis ---
class Analyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    def get_sentiment(self, text):
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']

# --- Utility Functions ---
def count_sentences(text):
    try:
        return len(sent_tokenize(text))
    except Exception as e:
        log_punkt_tab_error("count_sentences")
        # Fallback: count periods as a rough approximation
        return text.count('.')

def count_words(text):
    try:
        return len(word_tokenize(text))
    except Exception as e:
        log_punkt_tab_error("count_words")
        # Fallback: split by whitespace
        return len(text.split())

# --- Metadata Enrichment Functions ---
def detect_chunk_type(text):
    pain_point_keywords = [
        'challenge', 'challenges', 'struggled', 'struggle', 'difficult', 'difficulty', 'problem', 'problems',
        'pain', 'pains', 'pain point', 'pain points', 'issue', 'issues', 'obstacle', 'obstacles'
    ]
    if any(word in text.lower() for word in pain_point_keywords):
        return "problem"
    return "narrative"

def detect_themes(text):
    themes = []
    if "customer" in text.lower():
        themes.append("customer engagement")
    if "automation" in text.lower() or "data" in text.lower():
        themes.append("automation")
    return themes

def extract_pain_points(text):
    pain_point_keywords = [
        'challenge', 'challenges', 'struggled', 'struggle', 'difficult', 'difficulty', 'problem', 'problems',
        'pain', 'pains', 'pain point', 'pain points', 'issue', 'issues', 'obstacle', 'obstacles'
    ]
    found = [word for word in pain_point_keywords if word in text.lower()]
    return list(set(found))

def extract_needs(text):
    need_keywords = [
        'need', 'needs', 'needed', 'needing', 'require', 'requires', 'required', 'requiring', 'requirement',
        'requirements', 'want', 'wants', 'wanted', 'wanting', 'desire', 'desires', 'desired', 'desiring'
    ]
    found = [word for word in need_keywords if word in text.lower()]
    return list(set(found))

def extract_features(text):
    features = []
    if "ChurnZero" in text:
        features.append("ChurnZero")
    return features

def extract_personas(text):
    personas = []
    if "manager" in text.lower():
        personas.append("Manager")
    if "ceo" in text.lower():
        personas.append("CEO")
    return personas

def extract_metrics(text):
    return re.findall(r'[\$%\d]+', text)

def contains_quote(text):
    return '"' in text or "\"" in text or "\"" in text

def enrich_metadata(chunk_text):
    return {
        "chunk_type": detect_chunk_type(chunk_text),
        "themes": detect_themes(chunk_text),
        "detected_pain_points": extract_pain_points(chunk_text),
        "detected_needs": extract_needs(chunk_text),
        "features_used": extract_features(chunk_text),
        "personas": extract_personas(chunk_text),
        "metrics": extract_metrics(chunk_text),
        "quote": contains_quote(chunk_text)
    }

# --- Chunking Function for YouTube Transcripts ---
def chunk_youtube_transcript_by_chars(text, min_chars=900, max_chars=1300):
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        log_punkt_tab_error("chunk_youtube_transcript")
        # Fallback to splitting by newlines and then periods
        sentences = []
        for paragraph in text.split("\n"):
            if paragraph.strip():
                # Crude sentence splitting by periods followed by space
                for sent in re.split(r'\. +', paragraph):
                    if sent.strip():
                        sentences.append(sent.strip() + ".")
    
    # If we still don't have sentences, split by newlines
    if len(sentences) == 1:
        sentences = [s for s in text.split("\n") if s.strip()]
    
    chunks = []
    current_chunk = ""
    previous_sentiment = None
    
    # Instantiate the analyzer
    analyzer = Analyzer()

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        current_sentiment = analyzer.get_sentiment(sentence)
        
        if not current_chunk:
            current_chunk = sentence
            previous_sentiment = current_sentiment
            continue
            
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            if abs(current_sentiment - previous_sentiment) > 0.2 and len(current_chunk) >= min_chars:
                chunks.append(current_chunk)
                current_chunk = sentence
                previous_sentiment = current_sentiment
            else:
                current_chunk += " " + sentence
                previous_sentiment = current_sentiment
        else:
            if len(current_chunk) >= min_chars:
                chunks.append(current_chunk)
                current_chunk = sentence
                previous_sentiment = current_sentiment
            else:
                current_chunk += " " + sentence
                chunks.append(current_chunk)
                current_chunk = ""
                previous_sentiment = None
                
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def run_module_2_content_chunking(input_json_path, output_chunk_path):
    """Module 2: Content Chunking for Embeddings
    Processes the retrieved content by chunking it into smaller pieces suitable for embeddings,
    enriches the chunks with metadata, and saves the results in a second JSON file.
    
    Args:
        input_json_path: Path to the JSON file generated by Module 1
        output_chunk_path: Path where the chunked output will be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("MODULE 2: CONTENT CHUNKING FOR EMBEDDINGS")
    print("=" * 80)
    
    # Debug - Print input and output paths
    print(f"Debug - Module 2 received paths:")
    print(f"  Input JSON: {input_json_path}")
    print(f"  Output Chunk Path: {output_chunk_path}")
    
    module_start_time = datetime.now()
    
    try:
        # Read the input YouTube transcripts JSON file
        with open(input_json_path, 'r', encoding='utf8') as f:
            posts = json.load(f)

        enriched_posts = []
        print(f"Processing {len(posts)} YouTube transcripts for chunking...")

        # Instantiate the analyzer
        analyzer = Analyzer()

        for idx, post in enumerate(posts):
            title = post.get('title', f"YouTubeTranscript_{idx+1}")
            url = post.get('url', '')
            content = post.get('content', '')

            if not content:
                print(f"Skipping YouTube transcript #{idx+1} ({title}) - No content found")
                continue

            # Split content into chunks based on character length and sentiment shifts
            chunks = chunk_youtube_transcript_by_chars(content, min_chars=900, max_chars=1300)
            print(f"YouTube transcript #{idx+1} ({title}) - Split into {len(chunks)} chunks")

            total_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                # Enrich each chunk with metadata
                metadata = enrich_metadata(chunk)
                enriched_chunk = {
                    "id": f"{title.replace(' ', '_')}_{i+1}",
                    "youtube_transcript_title": title,
                    "youtube_transcript_url": url,
                    "chunk_id": i + 1,
                    "total_chunks": total_chunks,
                    "content": chunk,
                    "sentence_count": count_sentences(chunk),
                    "word_count": count_words(chunk),
                    "token_count": len(chunk.split()),  # Simplified token counting
                    "sentiment": analyzer.get_sentiment(chunk),
                    "processing_date": datetime.now().strftime("%Y-%m-%d"),
                    "script_version": "YOUTUBE-TRANSCRIPT-1.0"
                }
                enriched_chunk.update(metadata)
                enriched_posts.append(enriched_chunk)

        # Write the enriched YouTube transcripts to the output JSON file and upload to Supabase
        with open(output_chunk_path, 'w', encoding='utf8') as f:
            json.dump(enriched_posts, f, indent=2)
        
        # Upload to Supabase bucket
        with open(output_chunk_path, "rb") as f:
            data = f.read()
        res = supabase.storage.from_("agentic-output").upload(os.path.basename(output_chunk_path), data)
        if not res:
            raise Exception(f"Failed to upload {output_chunk_path} to Supabase bucket")
        print(f"Uploaded {os.path.basename(output_chunk_path)} to Supabase bucket")

        print("Processing complete! All YouTube transcripts have been processed and enriched.")
        print(f"Chunked output saved to {output_chunk_path}")
        print(f"Module 2 execution completed in {datetime.now() - module_start_time}")
        
        return True

    except Exception as error:
        print("Error processing the JSON file:", error)
        print(f"Module 2 failed after {datetime.now() - module_start_time}")
        return False

# =====================================================================
# MODULE 4: EMBEDDING GENERATION AND PINECONE UPSERT
# =====================================================================

# --- Helper Functions ---
def sanitize_id(id_string):
    """Sanitize ID strings to ensure they only contain Pinecone-compatible characters."""
    # Match exactly the pattern from the successful script
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', str(id_string))
    
    # CRITICAL FIX: Add a timestamp to ensure uniqueness
    # This prevents ID collisions with existing vectors
    timestamp = int(time.time())
    sanitized = f"{sanitized}_{timestamp}"
    
    # Ensure ID is not too long (Pinecone has limits)
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
        
    return sanitized

def get_embedding(text, api_key, model="text-embedding-3-small", max_retries=3, retry_delay=2):
    """Get embedding vector for text using OpenAI API with retry logic."""
    # LOGGING POINT 1: Log the embedding request details
    print(f"LOG-POINT-1: Generating embedding for text of length {len(text)} with model {model}")
    
    # Create a client instance with the API key - now passing api_key as parameter
    client = openai.OpenAI(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            # Use the client instance and pass text as a list - CRITICAL DIFFERENCE
            response = client.embeddings.create(input=[text], model=model)
            embedding = response.data[0].embedding
            
            # Verify the embedding is not all zeros
            if all(v == 0 for v in embedding):
                print("WARNING: Received all-zero embedding from OpenAI. Adding small random values.")
                embedding = [v + random.uniform(0.000001, 0.00001) for v in embedding]
            return embedding
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Embedding API error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to get embedding after {max_retries} attempts: {e}")
                raise

def run_module_4_embedding_generation(input_processed_path, output_embeddings_path):
    """Module 4: Embedding Generation and Pinecone Upsert
    Generates embeddings for each chunk, upserts the embeddings and metadata to Pinecone vector database,
    and saves a complete record of all processed data.
    
    Args:
        input_processed_path: Path to the processed JSON file generated by Module 3
        output_embeddings_path: Path where the embeddings output will be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("MODULE 4: EMBEDDING GENERATION AND PINECONE UPSERT")
    print("=" * 80)
    
    module_start_time = datetime.now()
    
    # Load environment variables
    # Environment variables already loaded in global section
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    supabase_url = os.getenv("VITE_SUPABASE_URL")
    # Supabase credentials are already loaded in the global section

    # LOGGING POINT 2: Log API key status (masked for security)
    print(f"LOG-POINT-2: OpenAI API key loaded: {bool(openai_api_key)} (length: {len(openai_api_key) if openai_api_key else 0})")
    print(f"LOG-POINT-2: Pinecone API key loaded: {bool(pinecone_api_key)} (length: {len(pinecone_api_key) if pinecone_api_key else 0})")

    if not openai_api_key or not pinecone_api_key:
        print("ERROR: Required API keys not found in environment file")
        return False

    # Fetch INDEX_NAME from Supabase (this is the only place it will be found)
    if not supabase_url or not supabase_service_key:
        print("ERROR: Supabase credentials not available in environment file")
        return False

    print("Attempting to fetch INDEX_NAME from Supabase...")
    try:
        # Fetch the most recent configuration
        variables = fetch_configuration_from_supabase(supabase_url, supabase_service_key)
        
        # Extract INDEX_NAME from the global section
        if "global" in variables and "INDEX_NAME" in variables["global"]:
            index_name = variables["global"]["INDEX_NAME"]
            print(f"Successfully fetched INDEX_NAME from Supabase: {index_name}")
        else:
            raise Exception("INDEX_NAME not found in Supabase config")
    except Exception as e:
        print(f"ERROR: Failed to fetch Pinecone index name from Supabase: {str(e)}")
        return False

    # Initialize Pinecone
    print(f"Initializing Pinecone with index '{index_name}'...")
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists
    try:
        available_indexes = pc.list_indexes().names()
        print(f"Available Pinecone indexes: {available_indexes}")
        
        if index_name not in available_indexes:
            print(f"ERROR: Index '{index_name}' not found in available indexes: {available_indexes}")
            return False
            
        # Connect to the index
        print(f"Connecting to Pinecone index '{index_name}'...")
        index = pc.Index(index_name)
    except Exception as e:
        print(f"ERROR: Failed to connect to Pinecone index: {str(e)}")
        return False

    # Verify connection and get initial vector count
    stats = index.describe_index_stats()
    initial_vector_count = stats.get('total_vector_count', 0)
    print(f"Connected to Pinecone index '{index_name}'. Current stats: {stats}")

    # Load and Process JSON
    print(f"Loading JSON from {input_processed_path}...")
    try:
        with open(input_processed_path, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"ERROR: Input file {input_processed_path} not found")
        return False

    try:
        with open(input_processed_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            chunk_count = len(chunks)

        print(f"JSON loaded successfully. Found {chunk_count} chunks to process.")

        # Process YouTube Transcript Chunks
        print("Processing YouTube transcript chunks and generating embeddings...")
        records = []
        total_chunks = chunk_count

        for i, chunk in enumerate(chunks):
            try:
                # Extract text content
                text = chunk.get("content", "").strip()
                if not text:
                    print(f"Skipping chunk {i} - empty content")
                    continue

                # LOGGING POINT 3: Log chunk details before embedding
                print(f"LOG-POINT-3: Processing chunk {i}, text length: {len(text)}, ID: {chunk.get('id', 'unknown')}")

                # Generate embedding - PASS THE API KEY
                embedding = get_embedding(text, openai_api_key)

                # Create record with proper metadata structure
                sanitized_id = sanitize_id(chunk.get("id", f"youtube_transcript_chunk_{i}"))
                
                # LOGGING POINT 4: Log embedding details
                print(f"LOG-POINT-4: Embedding generated for chunk {i}, embedding length: {len(embedding)}, sanitized ID: {sanitized_id}")
                
                record = {
                    "id": sanitized_id,
                    "text": text,
                    "embedding": embedding,
                    "metadata": {
                        # Include all available metadata fields
                        "chunk_id": chunk.get("chunk_id", ""),
                        "total_chunks": chunk.get("total_chunks", 0),
                        "youtube_transcript_title": chunk.get("youtube_transcript_title", ""),
                        "youtube_transcript_url": chunk.get("youtube_transcript_url", ""),
                        "chunk_type": chunk.get("chunk_type", ""),
                        "sentence_count": chunk.get("sentence_count", 0),
                        "word_count": chunk.get("word_count", 0),
                        "token_count": chunk.get("token_count", 0),
                        "sentiment": chunk.get("sentiment", 0),
                        "processing_date": chunk.get("processing_date", ""),
                        "script_version": chunk.get("script_version", ""),
                        "themes": chunk.get("themes", []),
                        "detected_pain_points": chunk.get("detected_pain_points", []),
                        "detected_needs": chunk.get("detected_needs", []),
                        "features_used": chunk.get("features_used", []),
                        "personas": chunk.get("personas", []),
                        "metrics": chunk.get("metrics", []),
                        "quote": chunk.get("quote", False)
                    }
                }

                records.append(record)

                # Print progress every 10 records
                if (i + 1) % 10 == 0 or i == total_chunks - 1:
                    print(f"Processed {i + 1}/{total_chunks} chunks")

            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                # Continue with next chunk rather than failing entire process

        print(f"Successfully processed {len(records)}/{total_chunks} chunks")

        # Save Processed Records to a JSON File and upload to Supabase bucket
        print(f"Saving processed records to {output_embeddings_path}...")
        with open(output_embeddings_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)
        
        # Upload to Supabase bucket
        with open(output_embeddings_path, "rb") as f:
            data = f.read()
        res = supabase.storage.from_("agentic-output").upload(os.path.basename(output_embeddings_path), data)
        if not res:
            raise Exception(f"Failed to upload {output_embeddings_path} to Supabase bucket")
        print(f"✅ Uploaded {os.path.basename(output_embeddings_path)} to Supabase bucket")

        # Insert Embeddings into Pinecone
        print("Inserting embeddings into Pinecone...")
        batch_size = 100
        total_batches = (len(records) + batch_size - 1) // batch_size
        successful_inserts = 0

        for i in range(0, len(records), batch_size):
            try:
                batch = records[i:i + batch_size]  # Extract batch
                batch_num = i // batch_size + 1

                # Format for Pinecone upsert
                vectors = []
                for r in batch:
                    vectors.append({
                        "id": r["id"],
                        "values": r["embedding"],
                        "metadata": r["metadata"]
                    })

                # Insert into Pinecone - with detailed logging
                try:
                    # LOGGING POINT 5: Log vector details before upsert
                    print(f"LOG-POINT-5: Upserting batch {batch_num}, vectors: {len(vectors)}, first vector ID: {vectors[0]['id'] if vectors else 'none'}")
                    print(f"LOG-POINT-5: First vector dimension: {len(vectors[0]['values']) if vectors and 'values' in vectors[0] else 'unknown'}")
                    
                    # CRITICAL FIX: Explicitly specify namespace as an empty string
                    # This ensures vectors go to the default namespace that's visible in stats
                    response = index.upsert(vectors=vectors, namespace="")
                    print(f"LOG-POINT-5: Upsert response: {response}")
                    successful_inserts += len(vectors)
                    
                    # Print progress
                    print(f"Inserted batch {batch_num}/{total_batches} ({len(vectors)} vectors)")
                except Exception as upsert_error:
                    print(f"ERROR during upsert operation: {str(upsert_error)}")
                    # Try with a smaller batch if this fails
                    if len(vectors) > 10:
                        print("Attempting to upsert with smaller batches...")
                        for mini_batch_idx in range(0, len(vectors), 10):
                            mini_batch = vectors[mini_batch_idx:mini_batch_idx + 10]
                            try:
                                mini_response = index.upsert(vectors=mini_batch)
                                successful_inserts += len(mini_batch)
                                print(f"Inserted mini-batch {mini_batch_idx//10 + 1} ({len(mini_batch)} vectors)")
                            except Exception as mini_error:
                                print(f"ERROR during mini-batch upsert: {str(mini_error)}")

                # Small delay to avoid rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"Error inserting batch {batch_num}/{total_batches}: {e}")
                # Continue with next batch rather than failing the entire process

        print(f"Completed Pinecone insertion. Inserted {successful_inserts}/{len(records)} vectors")

        # Verify Pinecone Storage - with detailed logging
        print("Verifying Pinecone storage...")
        # CRITICAL FIX: Wait longer to ensure Pinecone has updated its stats
        print("Waiting 10 seconds for Pinecone to update stats...")
        time.sleep(10)
        
        try:
            # LOGGING POINT 6: Log detailed verification process
            print(f"LOG-POINT-6: Starting verification with {successful_inserts} vectors inserted")
            print(f"LOG-POINT-6: Initial vector count was {initial_vector_count}")
            print(f"LOG-POINT-6: Expected final count should be {initial_vector_count + successful_inserts}")
            
            # Get the stats directly without the dummy vector
            print("Getting final Pinecone index stats...")
            # CRITICAL FIX: Force a query to ensure stats are updated
            try:
                # Query a random vector to force stats refresh
                index.query(vector=[0.1]*1536, top_k=1, namespace="")
            except Exception as query_error:
                print(f"Query to refresh stats failed, but continuing: {query_error}")
                
            time.sleep(5)  # Give more time for stats to update
            
            # Now get the stats
            final_stats = index.describe_index_stats()
            actual_total_vectors = final_stats.get('total_vector_count', 0)
            print(f"LOG-POINT-6: Actual final count is {actual_total_vectors}")
        except Exception as stats_error:
            print(f"Error refreshing index stats: {stats_error}")
            # Fallback
            final_stats = index.describe_index_stats()
            actual_total_vectors = final_stats.get('total_vector_count', 0)
        
        expected_total_vectors = initial_vector_count + successful_inserts

        print(f"Total vectors in index according to Pinecone: {actual_total_vectors}")
        print(f"Expected total vectors (initial + inserted): {expected_total_vectors}")
        print(f"Pinecone index stats: {final_stats}")

        # Calculate execution time
        module_end_time = datetime.now()
        module_execution_time = module_end_time - module_start_time
        minutes, seconds = divmod(module_execution_time.total_seconds(), 60)

        # Print execution summary
        print("\n" + "=" * 50 + " EXECUTION SUMMARY " + "=" * 50)
        print(f"Input chunks processed: {total_chunks}")
        print(f"Successfully embedded chunks: {len(records)}")
        print(f"Vectors added to Pinecone: {successful_inserts}")
        print(f"Initial vector count: {initial_vector_count}")
        print(f"Final vector count (Pinecone): {actual_total_vectors}")
        print(f"Vectors added in this run: {successful_inserts}")
        print(f"Execution time: {int(minutes)} minutes and {seconds:.2f} seconds")
        print("=" * 120)
        print("=" * 5 + " SCRIPT EXECUTION COMPLETED " + "=" * 5)
        print("✅ Process completed successfully!")

        return True

    except Exception as e:
        print(f"ERROR in Module 4: {e}")
        return False

# =====================================================================
# MAIN FUNCTION
# =====================================================================

# Note: The global log_punkt_tab_error function is defined at the top of the file

def main():
    """Main function to orchestrate the entire workflow."""
    print("\n" + "=" * 80)
    print("COMBINED YOUTUBE TRANSCRIPT PROCESSING WORKFLOW")
    print("=" * 80)
    
    # Start timing the entire workflow
    workflow_start_time = datetime.now()
    
    # Load environment variables
    # Environment variables already loaded in global section
    supabase_url = os.getenv("VITE_SUPABASE_URL")
    # Supabase credentials are already loaded in the global section
    
    if not supabase_url or not supabase_service_key:
        print("ERROR: Supabase credentials not found in environment variables")
        print(f"Please ensure {ENV_FILE_PATH} contains VITE_SUPABASE_URL and VITE_SUPABASE_SERVICE_ROLE_KEY")
        return
    
    # Fetch configuration from Supabase
    try:
        print("Fetching configuration from Supabase...")
        variables = fetch_configuration_from_supabase(supabase_url, supabase_service_key)
        
        # Extract required configuration variables
        if "global" not in variables:
            print("ERROR: Global configuration section not found in Supabase")
            return
            
        global_config = variables["global"]
        company_name = global_config.get("COMPANY_NAME", "company")
        company_name_or_handle = global_config.get("COMPANY_NAME_OR_HANDLE", company_name)
        num_videos = int(variables["scripts"]["transcripts"].get("NUM_VIDEOS", 10))
        
        # Generate output file paths with date and time (YYYYMMDD_HHMMSS)
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use lowercase for company_name to maintain consistency with existing files
        company_name_lower = company_name.lower()
        
        # Get the actual number of transcripts from the transcript_config
        transcript_config = variables["scripts"]["transcripts"]
        actual_num_videos = int(transcript_config.get("NUM_VIDEOS", num_videos))
        
        # Define filenames for Supabase bucket (no directory path)
        youtube_json_path = f"{company_name_lower}_{actual_num_videos}_youtube_transcripts_{date_time}.json"
        chunk_json_path = f"{company_name_lower}_{actual_num_videos}_youtube_transcripts_chunked_{date_time}.json"
        embeddings_json_path = f"{company_name_lower}_{actual_num_videos}_youtube_transcripts_embeddings_{date_time}.json"
        
        print(f"Output file paths:")
        print(f"  YouTube Transcript JSON: {youtube_json_path}")
        print(f"  Chunk JSON: {chunk_json_path}")
        print(f"  Embeddings JSON: {embeddings_json_path}")
        
        print(f"Configuration loaded successfully for {company_name}")
        print(f"Will process up to {num_videos} YouTube video transcripts for channel/handle '{company_name_or_handle}'")
        print(f"Output files will be uploaded to Supabase 'agentic-output' bucket")
        
        # Execute Module 1: YouTube Transcript Scraping
        youtube_json_path = run_module_1_youtube_transcript_scraping(variables, youtube_json_path)
        if not youtube_json_path:
            print("ERROR: Module 1 failed. Stopping workflow.")
            return
            
        # Execute Module 2: Content Chunking
        success = run_module_2_content_chunking(youtube_json_path, chunk_json_path)
        if not success:
            print("ERROR: Module 2 failed. Stopping workflow.")
            return
            
        # Note: Module 3 is integrated into Module 2 in this implementation
        # So we proceed directly to Module 4
            
        # Execute Module 4: Embedding Generation and Pinecone Upsert
        success = run_module_4_embedding_generation(chunk_json_path, embeddings_json_path)
        if not success:
            print("ERROR: Module 4 failed.")
            return
            
        # Calculate total execution time
        workflow_end_time = datetime.now()
        workflow_execution_time = workflow_end_time - workflow_start_time
        minutes, seconds = divmod(workflow_execution_time.total_seconds(), 60)
        
        print("\n" + "=" * 80)
        print("WORKFLOW EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Company: {company_name}")
        print(f"YouTube transcripts processed: {actual_num_videos}")
        print(f"Channel/Handle: {company_name_or_handle}")
        print(f"Total execution time: {int(minutes)} minutes and {seconds:.2f} seconds")
        print(f"Output files (saved locally and uploaded to Supabase bucket 'agentic-output'):")
        print(f"  - YouTube transcript content: {os.path.basename(youtube_json_path)}")
        print(f"  - Chunked content: {os.path.basename(chunk_json_path)}")
        print(f"  - Embeddings: {os.path.basename(embeddings_json_path)}")
        print("=" * 80)
        print("\n✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"ERROR in workflow execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
