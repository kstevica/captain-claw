#!/usr/bin/env python3
"""
Script to fetch www.hr using BeautifulSoup and save content to pythonfetch.txt
"""

import requests
from bs4 import BeautifulSoup

def fetch_www_hr():
    url = "https://www.hr"
    
    try:
        print(f"Fetching content from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get text content (strip extra whitespace)
        text_content = soup.get_text(separator='\n', strip=True)
        
        # Save to file
        output_file = "pythonfetch.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        print(f"Content successfully saved to {output_file}")
        print(f"File size: {len(text_content)} characters")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_www_hr()
