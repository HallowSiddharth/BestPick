import requests
from bs4 import BeautifulSoup

contents = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "accept-language": "en-IN,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Ch-Ua-Platform": "Windows",
}
url = "https://www.amazon.in/dp/B0C45N5VPT"

response = requests.get(url, headers=contents)
soup = BeautifulSoup(response.text, "lxml")
title_element = soup.select_one("#productTitle")
print(title_element)
review_element = soup.select("div.review")
scrapedreviews = []
for review in review_element:
    content = review.select_one("span.review-text")
    scrapedreviews.append(content.text)
file = open("Scrape.txt", "w")
for i in scrapedreviews:
    if i[-9:] == "Read more":
        print(True)
        file.write(i[:-10])
    else:
        file.write(i[:-1])
file.close()
