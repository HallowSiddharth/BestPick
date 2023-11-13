import requests
from bs4 import BeautifulSoup
import time
import random
import emoji
import analyzer


def search(searchterm):
    words = searchterm.split()
    query = "+".join(words)
    request = f"https://www.amazon.in/s?k={query}"
    contents = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
        "accept-language": "en-IN,en;q=0.9",
    }
    response = requests.get(request, headers=contents)
    soup = BeautifulSoup(response.text, "lxml")
    asin_elements = soup.find_all("div", {"data-asin": True})
    asin_list = [i["data-asin"] for i in asin_elements if i["data-asin"] != ""]
    return asin_list


def extract_from_asin(asin):
    contents = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
        "accept-language": "en-IN,en;q=0.9",
    }

    time.sleep(random.uniform(1.0, 2.0))
    url = f"https://www.amazon.in/dp/{asin}"

    response = requests.get(url, headers=contents)
    soup = BeautifulSoup(response.text, "lxml")
    title_element = soup.select_one("#productTitle")
    # print(title_element)
    review_element = soup.select("div.review")
    scrapedreviews = []
    for review in review_element:
        content = review.select_one("span.review-text")
        scrapedreviews.append(content.text)
    final = []
    for review in scrapedreviews:
        rev = "".join(c for c in review if c not in emoji.EMOJI_DATA)
        final.append(rev)
    return title_element.contents[0], final


def process_query(query):
    asins = search(query)
    time.sleep(random.uniform(0.2, 1.5))
    master_reviews = {}
    for asin in asins:
        name, reviews = extract_from_asin(asin)
        master_reviews[(asin, name)] = reviews
    return master_reviews


def generate_score(reviews):
    pass


# if "__name__" == "__main__":
query = input("Query: ")
d = process_query(query)
answers = []
for product in d:
    asin = product[0]
    title = product[1]
    reviews = d[product]
    score = analyzer.save_training_model(reviews, r"dataset2.csv")
    answers.append([score, title])

answers.sort()

print("Best:", answers[-1][:15])
print()
print("Second:", answers[-2][:15])
print()
print("Third:", answers[-3][:15])
