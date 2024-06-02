import requests
from bs4 import BeautifulSoup
import json

def scrape_h1_and_p_tags(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the <h1> tag
        h1_tag = soup.find('h1')
        title = h1_tag.get_text().strip() if h1_tag else 'No title found'
        
        # Extract all <p> tags and merge their text
        p_tags = soup.find_all('p')
        paragraphs = ' '.join(p.get_text().strip() for p in p_tags)
        
        results = {
            'title': title,
            'content': paragraphs
        }
        
        # Save results to a JSON file
        with open('h1_and_p_tags.json', 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)
        
        print("Scraped data has been successfully saved to 'h1_and_p_tags.json'")
    else:
        print("Failed to retrieve the webpage. Status code:", response.status_code)


# Example usage
url = 'https://www.bbc.com/telugu/india-50715656'
scrape_h1_and_p_tags(url)
