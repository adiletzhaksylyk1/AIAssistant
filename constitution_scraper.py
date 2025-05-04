import requests
from bs4 import BeautifulSoup
import os

CONSTITUTION_URL = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
OUTPUT_FILE = "temp_docs/constitution.txt"


def scrape_constitution():
    """
    Scrape the Constitution of the Republic of Kazakhstan from the official website
    and save it to a text file.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        print(f"Fetching constitution from {CONSTITUTION_URL}...")
        response = requests.get(CONSTITUTION_URL)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract content (this will need to be adjusted based on the actual HTML structure)
        # Typical approach is to find the main content div/section
        content_div = soup.find('div', class_='content') or soup.find('main') or soup.find('article')

        if content_div:
            # Extract text, preserving paragraph structure
            paragraphs = content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            content = "\n\n".join([p.get_text().strip() for p in paragraphs])

            # Save to file
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Constitution saved to {OUTPUT_FILE}")
            return True
        else:
            print("Could not find content section in the webpage")
            # Create a note in the file about the issue
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write("Could not automatically extract the Constitution text. Please manually download it.")
            return False

    except Exception as e:
        print(f"Error scraping constitution: {e}")
        return False


if __name__ == "__main__":
    scrape_constitution()