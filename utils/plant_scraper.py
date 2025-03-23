import requests
from bs4 import BeautifulSoup

def scrape_top_plants(url="https://example.com/top-plants"):
    """
    Scrape top recommended plants from the given URL.
    Returns a list of at most 3 plant recommendations with minimal details.
    """
    response = requests.get(url)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    recommendations = []
    
    # Example: Assume the website contains plant info in <div class="plant">
    for plant_div in soup.find_all("div", class_="plant")[:3]:
        name = plant_div.find("h2").get_text(strip=True)
        description = plant_div.find("p").get_text(strip=True)
        recommendations.append({
            "name": name,
            "description": description
        })
    return recommendations

# For testing purposes:
if __name__ == "__main__":
    plants = scrape_top_plants()
    print("Scraped plants:", plants)
