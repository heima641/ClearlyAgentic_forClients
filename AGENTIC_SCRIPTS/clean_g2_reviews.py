#!/usr/bin/env python3
"""
JSON Reformatter Script

This script reformats JSON files from the kluster_all-reviews format to match the 
churnzero_reviews format, adding a sixth field for "problems solved".

Mapping:
- publish_date (without timestamp) -> Review Date
- reviewer.reviewer_job_title -> Reviewer Role
- reviewer_company_size -> Company Size
- Answer to "What do you like best about Kluster?" -> like
- Answer to "What do you dislike about Kluster?" -> dislike
- Answer to "What problems is Kluster solving and how is that benefiting you?" -> problems solved
"""

import json
import argparse
from datetime import datetime
import os

def reformat_json(input_file, output_file):
    """
    Reformat the JSON from kluster_all-reviews format to churnzero_reviews format.
    
    Args:
        input_file (str): Path to the input JSON file (kluster format)
        output_file (str): Path to the output JSON file (churnzero format)
    """
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Initialize the output data as an empty list
        output_data = []
        
        # Process each review in the input data
        for review in input_data[0]['all_reviews']:
            # Extract the required fields
            
            # Format the publish date (remove timestamp)
            publish_date = review.get('publish_date', '')
            if publish_date:
                # Convert from ISO format (2024-11-06T00:00:00) to MM/DD/YYYY
                try:
                    date_obj = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime('%m/%d/%Y')
                except ValueError:
                    formatted_date = publish_date.split('T')[0]  # Fallback to just removing the T part
            else:
                formatted_date = ''
            
            # Get reviewer job title
            reviewer = review.get('reviewer', {})
            reviewer_role = reviewer.get('reviewer_job_title', '')
            
            # Get company size
            company_size = review.get('reviewer_company_size', '')
            
            # Extract answers to specific questions
            like_text = ''
            dislike_text = ''
            problems_solved_text = ''
            
            for qa in review.get('review_question_answers', []):
                question = qa.get('question', '').lower()
                answer = qa.get('answer', '')
                
                if 'like best' in question:
                    like_text = answer
                elif 'dislike' in question:
                    dislike_text = answer
                elif 'problems' in question and 'solving' in question:
                    problems_solved_text = answer
            
            # Create the reformatted review
            reformatted_review = {
                "Review Date": formatted_date,
                "Reviewer Role": reviewer_role,
                "Company Size": company_size,
                "like": like_text,
                "dislike": dislike_text,
                "problems solved": problems_solved_text
            }
            
            # Add to the output data
            output_data.append(reformatted_review)
        
        # Write the output data to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully reformatted {len(output_data)} reviews from {input_file} to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error reformatting JSON: {e}")
        return False

def main():
    """Main function to parse arguments and call the reformat function."""
    parser = argparse.ArgumentParser(description='Reformat JSON files from kluster format to churnzero format')
    parser.add_argument('input_file', help='Path to the input JSON file (kluster format)')
    parser.add_argument('output_file', help='Path to the output JSON file (churnzero format)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return
    
    # Reformat the JSON
    reformat_json(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
