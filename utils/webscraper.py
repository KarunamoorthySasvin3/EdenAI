# filepath: utils/webscraper.py
import requests
from bs4 import BeautifulSoup
import json

def fetch_plant_info(plant_name: str) -> dict:
    url = f"https://example.com/plants/{plant_name.replace(' ', '-').lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract relevant details depending on the website structure
        description = soup.find("div", class_="plant-description").text.strip()
        care = soup.find("div", class_="plant-care").text.strip()
        return {"name": plant_name, "description": description, "care": care}
    return {"name": plant_name, "description": "Not found", "care": "Not found"}

def scrape_plant_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract plant data (adapt to the website structure)
    plant_name = soup.find('h1').text.strip()
    sunlight = soup.find('div', {'class': 'sunlight'}).text.strip()
    water = soup.find('div', {'class': 'water'}).text.strip()
    description = soup.find("div", class_="plant-description").text.strip()
    # ... extract other features
    
    plant_data = {
        'name': plant_name,
        'sunlight': sunlight,
        'water': water,
        'description': description,
        # ... other features
    }
    return plant_data

def main():
    plant_list = []
    # List of URLs to scrape
    urls = [
        'https://example.com/plant1',
        'https://example.com/plant2',
        # ... add more URLs
    ]
    
    for url in urls:
        plant_data = scrape_plant_data(url)
        plant_list.append(plant_data)
    
    # Save to JSON file
    with open('data/plants.json', 'w') as f:
        json.dump(plant_list, f, indent=4)

if __name__ == "__main__":
    main()