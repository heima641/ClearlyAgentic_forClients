#!/usr/bin/env python3
"""
Standard Question Analyzer Script v2 + Poppy Card Generator

Automatically reads the most recent COMPANY_NAME from Supabase workflow_configs,
generates structured problem/solution analysis, saves both JSON and TXT files 
to the agentic-output Supabase bucket, then generates 15 Poppy cards and saves
them to the poppy-cards Supabase bucket. Designed for n8n workflow integration.
"""

import os
import json
import sys
import re
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import openai
from pinecone import Pinecone
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class StructuredLogger:
    def __init__(self):
        self.start_time = datetime.now()
        
    def log(self, level: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Structured logging for n8n monitoring"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds()
        }
        if data:
            log_entry["data"] = data
            
        print(f"[{level}] {timestamp} - {message}")
        if data:
            print(f"    Data: {json.dumps(data, indent=2)}")
        
        return log_entry
        
    def success(self, message: str, data: Optional[Dict[str, Any]] = None):
        return self.log("SUCCESS", message, data)
        
    def error(self, message: str, data: Optional[Dict[str, Any]] = None):
        return self.log("ERROR", message, data)
        
    def info(self, message: str, data: Optional[Dict[str, Any]] = None):
        return self.log("INFO", message, data)
        
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        return self.log("WARNING", message, data)

class StandardQuestionAnalyzer:
    def __init__(self):
        """Initialize the analyzer with API credentials and configurations."""
        self.logger = StructuredLogger()
        
        # API Keys - validate they exist (check both naming conventions)
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.supabase_url = os.getenv('SUPABASE_URL') or os.getenv('VITE_SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY') or os.getenv('VITE_SUPABASE_ANON_KEY')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('VITE_SUPABASE_SERVICE_ROLE_KEY')
        
        if not all([self.openai_api_key, self.pinecone_api_key, self.supabase_url, self.supabase_key]):
            self.logger.error("Missing required API credentials", {
                "openai_key_present": bool(self.openai_api_key),
                "pinecone_key_present": bool(self.pinecone_api_key),
                "supabase_url_present": bool(self.supabase_url),
                "supabase_key_present": bool(self.supabase_key)
            })
            raise ValueError("Missing required API credentials")
        
        # Initialize clients
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
        self.supabase_client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize storage client with service role key if available
        if self.supabase_service_key:
            self.supabase_storage_client = create_client(self.supabase_url, self.supabase_service_key)
            self.logger.info("Using service role key for storage operations")
        else:
            self.supabase_storage_client = self.supabase_client
            self.logger.info("Using anon key for storage operations (service role key not available)")
        
        self.logger.info("Initialized StandardQuestionAnalyzer successfully")
        
        # Standard question prompt
        self.standard_prompt = """Please list 7 (relevant and representative) customer problem statements and explain how the product helped to solve each problem. Provide 8 (eight) customer quotes for each of the 7 problem/solution statements. Do not only provide 2 quotes. Provide 8 (eight) quotes for each problem/solution statement. Provide an insightful 80-word business-focused summary of the ideal customer profile (ICP) based on the Pinecone database namespace content. _ The source database context provides ample information to identify distinct relevant and representative problem/solution statements with corresponding customer quotes. Pull insights from the Pinecone index and namespace."""
        
    def get_current_namespace_and_index(self) -> tuple[str, str]:
        """Get the most recent COMPANY_NAME and INDEX_NAME from Supabase workflow_configs."""
        try:
            self.logger.info("Fetching most recent workflow configuration from Supabase")
            
            # Query for the most recent config entry
            response = self.supabase_client.table('workflow_configs')\
                .select('variables')\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if not response.data:
                raise ValueError("No workflow configurations found in Supabase")
            
            config_data = response.data[0]['variables']
            global_config = config_data.get('global', {})
            
            company_name = global_config.get('COMPANY_NAME')
            index_name = global_config.get('INDEX_NAME')
            
            if not company_name or not index_name:
                raise ValueError("COMPANY_NAME or INDEX_NAME not found in most recent config")
            
            self.logger.success("Retrieved current configuration", {
                "company_name": company_name,
                "index_name": index_name
            })
            
            return company_name, index_name
            
        except Exception as e:
            self.logger.error(f"Failed to fetch current configuration: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for the given text using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def query_pinecone(self, namespace: str, index_name: str, query_text: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """Query Pinecone vector database for relevant content chunks."""
        try:
            self.logger.info(f"Querying Pinecone index '{index_name}' with namespace '{namespace}'")
            
            # Connect to the index
            index = self.pinecone_client.Index(index_name)
            
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query_text)
            
            # Query parameters
            query_params = {
                "vector": query_embedding,
                "include_metadata": True,
                "include_values": False,
                "top_k": top_k,
                "namespace": namespace
            }
            
            # Execute query
            response = index.query(**query_params)
            
            # Extract relevant content
            results = []
            for match in response.matches:
                if match.metadata:
                    results.append({
                        'content': match.metadata.get('text', ''),
                        'score': match.score,
                        'metadata': match.metadata
                    })
            
            self.logger.success(f"Retrieved {len(results)} content chunks from Pinecone")
            return results
            
        except Exception as e:
            self.logger.error(f"Pinecone query failed: {str(e)}")
            raise
    
    def format_context_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """Format Pinecone results into context for LLM processing."""
        context_parts = []
        
        # Debug: Log the first result to see the metadata structure
        if results:
            self.logger.info("Sample Pinecone result metadata keys", {
                "metadata_keys": list(results[0].get('metadata', {}).keys()) if results[0].get('metadata') else [],
                "first_result_structure": {k: type(v).__name__ for k, v in results[0].items()}
            })
        
        for i, result in enumerate(results[:15]):  # Limit to top 15 results to stay within token limits
            metadata = result.get('metadata', {})
            
            # Extract content using the same approach as the working Q&A module
            content_parts = []
            for key, value in metadata.items():
                if value and isinstance(value, str) and value.strip():
                    content_parts.append(f"{key.upper()}: {value}")
                elif value and not isinstance(value, (dict, list)):
                    content_parts.append(f"{key.upper()}: {str(value)}")
            
            if content_parts:
                full_content = "\n\n".join(content_parts)
                context_parts.append(f"Content Chunk {i+1}:\n{full_content}\n")
        
        final_context = "\n".join(context_parts)
        self.logger.info(f"Formatted context length: {len(final_context)} characters")
        
        return final_context
    
    def generate_structured_response(self, context: str, namespace: str) -> str:
        """Generate the structured response using OpenAI with the provided context."""
        try:
            self.logger.info("Generating structured response with OpenAI")
            
            system_prompt = f"""You are an expert business analyst specializing in customer insights and problem-solution analysis. You have access to comprehensive customer feedback, reviews, and case studies from the Pinecone database namespace '{namespace}'.

CRITICAL FORMATTING REQUIREMENTS:
You MUST follow this EXACT format structure for compatibility with downstream processing:

**Problem 1: [Title]**
[Problem description and solution explanation]

**Quotes for Problem 1:**
1. "[Quote 1]"
2. "[Quote 2]"
3. "[Quote 3]"
4. "[Quote 4]"
5. "[Quote 5]"
6. "[Quote 6]"
7. "[Quote 7]"
8. "[Quote 8]"

**Problem 2: [Title]**
[Problem description and solution explanation]

**Quotes for Problem 2:**
1. "[Quote 1]"
2. "[Quote 2]"
3. "[Quote 3]"
4. "[Quote 4]"
5. "[Quote 5]"
6. "[Quote 6]"
7. "[Quote 7]"
8. "[Quote 8]"

[Continue this exact pattern for Problems 3, 4, 5, 6, and 7]

**Ideal Customer Profile (ICP):**
[Exactly 80 words summarizing the ideal customer profile]

FORMATTING RULES:
- Start each problem section with exactly: **Problem [NUMBER]: [TITLE]**
- Follow each problem with exactly: **Quotes for Problem [NUMBER]:**
- Number quotes 1-8 with exactly: [NUMBER]. "[QUOTE]"
- End with exactly: **Ideal Customer Profile (ICP):**
- Use double asterisks ** for all headers
- Include blank lines between sections
- No other formatting variations allowed

Requirements:
- Use only authentic content from the provided database context
- Ensure each problem statement is distinct and representative
- Each set of 8 quotes should directly relate to the specific problem/solution
- Quotes should be varied and demonstrate different aspects of the solution
- The ICP should synthesize insights from across all the content"""

            user_prompt = f"""Based on the following customer content from the '{namespace}' database:

{context}

{self.standard_prompt}

IMPORTANT: Follow the EXACT formatting structure specified in the system prompt. This format is required for downstream processing compatibility."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=3000,
                temperature=0.7
            )
            
            result = response.choices[0].message.content
            self.logger.success("Generated structured response successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"OpenAI response generation failed: {str(e)}")
            raise
    
    def create_text_summary(self, full_response: str, namespace: str) -> str:
        """Create a simplified text summary from the full response."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        summary = f"""STANDARD QUESTION ANALYSIS SUMMARY
Namespace: {namespace}
Generated: {timestamp}

{full_response}

---
Generated by Standard Question Analyzer
For technical details, see the corresponding JSON file.
"""
        return summary
    
    def upload_to_supabase_storage(self, content: str, filename: str, content_type: str, bucket: str = 'agentic-output') -> str:
        """Upload content to the specified Supabase storage bucket."""
        try:
            self.logger.info(f"Uploading {filename} to Supabase storage bucket '{bucket}'")
            
            # Upload to the specified bucket using storage client (with service role key if available)
            response = self.supabase_storage_client.storage.from_(bucket).upload(
                filename, 
                content.encode('utf-8'),
                file_options={
                    'content-type': content_type,
                    'upsert': 'true'
                }
            )
            
            if response.path:
                # Get the public URL
                public_url = self.supabase_storage_client.storage.from_(bucket).get_public_url(filename)
                self.logger.success(f"Successfully uploaded {filename}", {
                    "bucket": bucket,
                    "filename": filename,
                    "public_url": public_url
                })
                return public_url
            else:
                raise Exception(f"Upload failed: {response}")
                
        except Exception as e:
            self.logger.error(f"Failed to upload {filename} to Supabase storage: {str(e)}")
            raise
    
    def download_from_supabase_storage(self, filename: str, bucket: str = 'agentic-output') -> str:
        """Download content from the specified Supabase storage bucket."""
        try:
            self.logger.info(f"Downloading {filename} from Supabase storage bucket '{bucket}'")
            
            # Download from the specified bucket
            response = self.supabase_storage_client.storage.from_(bucket).download(filename)
            
            if response:
                content = response.decode('utf-8')
                self.logger.success(f"Successfully downloaded {filename} from {bucket}")
                return content
            else:
                raise Exception(f"Download failed for {filename}")
                
        except Exception as e:
            self.logger.error(f"Failed to download {filename} from Supabase storage: {str(e)}")
            raise
    
    def run_analysis(self) -> Dict[str, str]:
        """Run the complete standard question analysis."""
        self.logger.info("Starting Standard Question Analysis")
        
        try:
            # Step 1: Get current namespace and index from most recent config
            namespace, index_name = self.get_current_namespace_and_index()
            
            # Step 2: Query Pinecone for relevant content
            self.logger.info("Querying Pinecone database for content")
            results = self.query_pinecone(namespace, index_name, self.standard_prompt)
            
            if not results:
                raise ValueError(f"No content found in namespace '{namespace}'")
            
            # Step 3: Format context for LLM
            context = self.format_context_for_llm(results)
            
            # Step 4: Generate structured response
            response = self.generate_structured_response(context, namespace)
            
            # Step 5: Create files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # JSON file with complete data
            json_data = {
                "namespace": namespace,
                "index_name": index_name,
                "timestamp": datetime.now().isoformat(),
                "query": self.standard_prompt,
                "response": response,
                "metadata": {
                    "total_chunks_retrieved": len(results),
                    "context_length": len(context),
                    "analysis_version": "v2"
                }
            }
            
            json_filename = f"{namespace}_standard_analysis_{timestamp}.json"
            json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            # Text summary file
            txt_filename = f"{namespace}_standard_analysis_{timestamp}.txt"
            txt_content = self.create_text_summary(response, namespace)
            
            # Step 6: Upload files to Supabase storage
            # Upload JSON file to Supabase storage
            json_url = self.upload_to_supabase_storage(json_content, json_filename, 'application/json')
            
            # Upload TXT file to Supabase storage
            txt_url = self.upload_to_supabase_storage(txt_content, txt_filename, 'text/plain')
            
            self.logger.success("Files uploaded to Supabase storage successfully", {
                "json_file": json_filename,
                "txt_file": txt_filename,
                "json_url": json_url,
                "txt_url": txt_url
            })
            
            result = {
                "namespace": namespace,
                "json_file": json_url,
                "txt_file": txt_url,
                "txt_filename": txt_filename,  # Add this for Poppy card generation
                "status": "SUCCESS"
            }
            
            self.logger.success("Analysis completed successfully", result)
            return result
            
        except Exception as e:
            error_result = {
                "status": "ERROR",
                "error": str(e)
            }
            self.logger.error("Analysis failed", error_result)
            return error_result

class PoppyCardGenerator:
    """Generates Poppy cards from standard analysis files."""
    
    def __init__(self, analyzer_instance):
        """Initialize with the analyzer instance to reuse logger and storage client."""
        self.logger = analyzer_instance.logger
        self.supabase_storage_client = analyzer_instance.supabase_storage_client
    
    def parse_standard_analysis_content(self, content: str) -> tuple[str, Dict[int, str]]:
        """Parse the standard analysis content to extract problem statement packages and namespace."""
        
        # Extract namespace from the beginning of the content
        namespace_match = re.search(r'Namespace:\s*(\w+)', content, re.IGNORECASE)
        if not namespace_match:
            raise ValueError("Could not find namespace in the input content. Expected format: 'Namespace: <name>'")
        
        namespace = namespace_match.group(1).lower()
        
        problems = {}
        
        # Look for the exact format we specified in the AI prompt: **Problem N: Title**
        problem_pattern = r'\*\*Problem\s+(\d+):'
        problem_matches = list(re.finditer(problem_pattern, content, re.IGNORECASE))
        
        if not problem_matches:
            # Fallback to other common patterns if the strict format isn't found
            fallback_patterns = [
                r'(?:###|##|\*\*)\s*(\d+)\.?\s*Problem[:\s]',
                r'\*?\*?Problem\s+(\d+):',
                r'^(\d+)\.?\s*(?:Problem|PROBLEM)[:\s]'
            ]
            
            for pattern in fallback_patterns:
                problem_matches = list(re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE))
                if problem_matches:
                    self.logger.warning(f"Using fallback pattern: {pattern}")
                    break
        
        if not problem_matches:
            raise ValueError("Could not find problem statements in the expected format. Expected: **Problem 1:** through **Problem 7:**")
        
        # Verify we have exactly 7 problems
        if len(problem_matches) != 7:
            found_problems = [int(match.group(1)) for match in problem_matches]
            raise ValueError(f"Expected exactly 7 problem statements, found {len(problem_matches)}. "
                            f"Problem numbers found: {sorted(found_problems)}")
        
        # Extract each problem package
        for i, match in enumerate(problem_matches):
            problem_num = int(match.group(1))
            start_pos = match.start()
            
            # Find end position (start of next problem or ICP section)
            if i + 1 < len(problem_matches):
                end_pos = problem_matches[i + 1].start()
            else:
                # For the last problem, look for ICP section
                icp_match = re.search(r'\*\*Ideal Customer Profile \(ICP\):\*\*', content[start_pos:], re.IGNORECASE)
                if icp_match:
                    end_pos = start_pos + icp_match.start()
                else:
                    # Fallback: look for any ICP marker or end of content
                    icp_fallback = re.search(r'(?:###|##|\*\*)\s*(?:Ideal Customer Profile|ICP)', content[start_pos:], re.IGNORECASE)
                    if icp_fallback:
                        end_pos = start_pos + icp_fallback.start()
                    else:
                        end_pos = len(content)
            
            # Extract the full problem package
            problem_package = content[start_pos:end_pos].strip()
            
            # Clean up the package - remove extra whitespace and normalize
            problem_package = re.sub(r'\n\s*\n\s*\n+', '\n\n', problem_package)
            problem_package = problem_package.strip()
            
            problems[problem_num] = problem_package
        
        # Verify we have problems 1-7
        expected_problems = set(range(1, 8))
        found_problems = set(problems.keys())
        missing_problems = expected_problems - found_problems
        
        if missing_problems:
            raise ValueError(f"Missing problem statement(s): {sorted(missing_problems)}. "
                            f"Found problems: {sorted(found_problems)}")
        
        self.logger.success(f"Successfully parsed {len(problems)} problem sections using structured format")
        return namespace, problems
    
    def generate_poppy_cards(self, namespace: str, problems: Dict[int, str]) -> List[str]:
        """Generate 15 Poppy card files and upload to Supabase storage."""
        
        self.logger.info("Starting Poppy card generation")
        
        # Define the 15 card specifications
        card_specs = [
            ("card01", [1, 2, 3, 4]),
            ("card02", [2, 3, 4, 5]),
            ("card03", [3, 4, 5, 6]),
            ("card04", [4, 5, 6, 7]),
            ("card05", [1, 3, 5, 7]),
            ("card06", [2, 3, 6, 7]),
            ("card07", [6, 5, 2, 1]),
            ("card08", [7, 3, 2, 1]),
            ("card09", [5, 3, 6, 4]),
            ("card10", [2, 4, 6, 1]),
            ("card11", [1, 2]),
            ("card12", [3, 4]),
            ("card13", [5, 6]),
            ("card14", [2, 7]),
            ("card15", [4, 1])
        ]
        
        uploaded_files = []
        
        # Generate each card
        for card_name, problem_sequence in card_specs:
            # Create the P sequence for filename
            p_sequence = "_".join([f"P{p}" for p in problem_sequence])
            filename = f"{namespace}_{card_name}_{p_sequence}.txt"
            
            # Create header for the card
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"STANDARD QUESTION ANALYSIS SUMMARY for Card\nNamespace: {namespace}\nGenerated: {current_time}\n"
            
            # Build the card content
            card_content = []
            for problem_num in problem_sequence:
                if problem_num in problems:
                    card_content.append(problems[problem_num])
                else:
                    raise ValueError(f"Problem {problem_num} is required for {card_name} but not found in input file")
            
            # Join with blank lines between packages
            problem_content = "\n\n".join(card_content)
            
            # Combine header with problem content
            full_content = header + "\n" + problem_content
            
            # Upload to poppy-cards bucket
            try:
                public_url = self.upload_to_poppy_cards_bucket(full_content, filename)
                uploaded_files.append(filename)
                self.logger.info(f"Generated Poppy card: {filename}")
                
            except Exception as e:
                self.logger.error(f"Failed to upload Poppy card {filename}: {str(e)}")
                raise
        
        self.logger.success(f"Generated {len(uploaded_files)} Poppy cards successfully")
        return uploaded_files
    
    def upload_to_poppy_cards_bucket(self, content: str, filename: str) -> str:
        """Upload content to the poppy-cards Supabase storage bucket."""
        try:
            # Upload to the poppy-cards bucket
            response = self.supabase_storage_client.storage.from_('poppy-cards').upload(
                filename, 
                content.encode('utf-8'),
                file_options={
                    'content-type': 'text/plain',
                    'upsert': 'true'
                }
            )
            
            if response.path:
                # Get the public URL
                public_url = self.supabase_storage_client.storage.from_('poppy-cards').get_public_url(filename)
                return public_url
            else:
                raise Exception(f"Upload failed: {response}")
                
        except Exception as e:
            raise Exception(f"Failed to upload {filename} to poppy-cards bucket: {str(e)}")
    
    def generate_cards_from_analysis_file(self, txt_filename: str) -> List[str]:
        """Generate Poppy cards from a standard analysis TXT file in Supabase storage."""
        try:
            # Download the TXT file content from agentic-output bucket
            self.logger.info(f"Downloading analysis file: {txt_filename}")
            content = self.download_from_agentic_output_bucket(txt_filename)
            
            # Parse the content to extract namespace and problems
            namespace, problems = self.parse_standard_analysis_content(content)
            
            # Generate the Poppy cards
            uploaded_files = self.generate_poppy_cards(namespace, problems)
            
            return uploaded_files
            
        except Exception as e:
            self.logger.error(f"Failed to generate Poppy cards from {txt_filename}: {str(e)}")
            raise
    
    def download_from_agentic_output_bucket(self, filename: str) -> str:
        """Download content from the agentic-output Supabase storage bucket."""
        try:
            # Download from the agentic-output bucket
            response = self.supabase_storage_client.storage.from_('agentic-output').download(filename)
            
            if response:
                content = response.decode('utf-8')
                return content
            else:
                raise Exception(f"Download failed for {filename}")
                
        except Exception as e:
            raise Exception(f"Failed to download {filename} from agentic-output bucket: {str(e)}")

def main():
    """Main function to run the standard question analyzer and Poppy card generator."""
    print("=" * 60)
    print("Standard Question Analyzer v2 + Poppy Card Generator")
    print("Automated Problem/Solution Analysis with Poppy Card Generation")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = StandardQuestionAnalyzer()
        
        # Run standard question analysis
        result = analyzer.run_analysis()
        
        # Check if analysis was successful
        if result.get("status") == "SUCCESS":
            print(f"\n🎉 STANDARD ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"Namespace: {result['namespace']}")
            print(f"JSON file: {result['json_file']}")
            print(f"TXT file: {result['txt_file']}")
            
            # Log transition to Poppy card operation
            analyzer.logger.info("Standard analysis completed successfully. Moving on to Poppy card operation.")
            print("\n📋 MOVING ON TO POPPY CARD OPERATION...")
            
            # Initialize Poppy card generator
            poppy_generator = PoppyCardGenerator(analyzer)
            
            # Generate Poppy cards from the analysis file
            uploaded_files = poppy_generator.generate_cards_from_analysis_file(result['txt_filename'])
            
            print(f"\n🎯 POPPY CARD GENERATION COMPLETED!")
            print(f"Generated {len(uploaded_files)} Poppy cards in 'poppy-cards' bucket")
            print("Poppy card files:")
            for filename in uploaded_files:
                print(f"  - {filename}")
            
            # Final success message
            print(f"\n✅ COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
            print(f"Standard analysis and {len(uploaded_files)} Poppy cards generated for namespace: {result['namespace']}")
            
            sys.exit(0)  # Success exit code for n8n
        else:
            print(f"\n❌ STANDARD ANALYSIS FAILED: {result.get('error', 'Unknown error')}")
            sys.exit(1)  # Error exit code for n8n
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {str(e)}")
        sys.exit(1)  # Error exit code for n8n

if __name__ == "__main__":
    main()