#!/usr/bin/env python3
"""
JSON Reformatter Script

This script reformats JSON files from the old original <CompanyName>_all-reviews format to add a sixth field for "problems solved".

Mapping:
- publish_date (without timestamp) -> Review Date
- reviewer.reviewer_job_title -> Reviewer Role
- reviewer_company_size -> Company Size
- Answer to "What do you like best about <CompanyName>?" -> like
- Answer to "What do you dislike about <CompanyName>?" -> dislike
- Answer to "What problems is <CompanyName> solving and how is that benefiting you?" -> problems solved
"""

import json
import argparse
from datetime import datetime
import os

def reformat_json(input_file: str, output_file: str) -> bool:
    """
    Reformat G2-review JSON into a six-field CSV-ready list.

    Accepts **either** of the two raw formats you now generate:
    1. Legacy Apify payload  →  [ { "all_reviews": [...] } ]
    2. RapidAPI payload      →  { "initial_reviews": [...] }

    The output keeps only the fields you care about and adds the
    “problems solved” column.

    Parameters
    ----------
    input_file  : str
        Path to the raw JSON file.
    output_file : str
        Path to write the cleaned JSON.

    Returns
    -------
    bool
        True on success, False on any exception.
    """
    try:
        # ------------------------------------------------------------------
        # 1) Load the raw JSON
        # ------------------------------------------------------------------
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        # ------------------------------------------------------------------
        # 2) Locate and combine reviews arrays from all sources
        # ------------------------------------------------------------------
        reviews = []
        
        # Get reviews from initial_reviews section if available
        if isinstance(input_data, dict) and "initial_reviews" in input_data:
            reviews.extend(input_data["initial_reviews"])          # RapidAPI
        
        # Get reviews from all_reviews section if available
        if isinstance(input_data, dict) and "all_reviews" in input_data:
            reviews.extend(input_data["all_reviews"])              # RapidAPI with all_reviews
        elif (
            isinstance(input_data, list)
            and input_data
            and "all_reviews" in input_data[0]
        ):
            reviews.extend(input_data[0]["all_reviews"])           # Apify
            
        # Check if we found any reviews
        if not reviews:
            raise ValueError("No reviews found in the input JSON")

        # ------------------------------------------------------------------
        # 3) Transform each review
        # ------------------------------------------------------------------
        output_data = []
        for review in reviews:

            # --- 3-a ▸ format date ----------------------------------------
            publish_date = review.get("publish_date", "")
            if publish_date:
                try:
                    dt = datetime.fromisoformat(publish_date.replace("Z", "+00:00"))
                    formatted_date = dt.strftime("%m/%d/%Y")
                except ValueError:
                    # Fallbacks for non-ISO or otherwise malformed dates
                    if "T" in publish_date:
                        formatted_date = publish_date.split("T")[0]
                    else:
                        # Try simple YYYY-MM-DD pattern → mm/dd/YYYY
                        try:
                            formatted_date = datetime.strptime(publish_date[:10], "%Y-%m-%d").strftime("%m/%d/%Y")
                        except ValueError:
                            # As a last resort keep the original string
                            formatted_date = publish_date
            else:
                formatted_date = ""

            # --- 3-b ▸ basic fields ---------------------------------------
            reviewer_role = review.get("reviewer", {}).get("reviewer_job_title", "")
            company_size  = review.get("reviewer_company_size", "")

            # --- 3-c ▸ answers to key questions ---------------------------
            like_text, dislike_text, problems_solved_text = "", "", ""
            for qa in review.get("review_question_answers", []):
                question = qa.get("question", "").lower()
                answer   = qa.get("answer", "")

                if "like best" in question:
                    like_text = answer
                elif "dislike" in question:
                    dislike_text = answer
                elif "problems" in question and ("solving" in question or "benefiting" in question):
                    problems_solved_text = answer

            # --- 3-d ▸ assemble cleaned row -------------------------------
            output_data.append(
                {
                    "Review Date"    : formatted_date,
                    "Reviewer Role"  : reviewer_role,
                    "Company Size"   : company_size,
                    "like"           : like_text,
                    "dislike"        : dislike_text,
                    "problems solved": problems_solved_text,
                }
            )

        # ------------------------------------------------------------------
        # 4) Write out the cleaned JSON file
        #     (create parent dirs if they don’t exist)
        # ------------------------------------------------------------------
        out_dir = os.path.dirname(output_file)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Reformatted {len(output_data)} reviews → {output_file}")
        return True

    except Exception as e:
        print(f"❌ Error reformatting JSON: {e}")
        return False

def main():
    """Main function to parse arguments and call the reformat function."""
    parser = argparse.ArgumentParser(description='Reformat JSON files from <CompanyName> format to problems solved sixth field format')
    parser.add_argument('input_file', help='Path to the input JSON file (<CompanyName> format)')
    parser.add_argument('output_file', help='Path to the output JSON file (problems solved sixth field format)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return
    
    # Reformat the JSON
    reformat_json(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
