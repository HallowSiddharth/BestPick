import requests
from bs4 import BeautifulSoup
import time

def search(searchterm):
    words = searchterm.split()
    query = "+".join(words)
    request = f"https://www.amazon.in/s?k={query}"
    contents = {"user-agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0', 'accept-language':'en-IN,en;q=0.9'}
    response = requests.get(url,headers=contents)
    soup = BeautifulSoup(response.text,'lxml')
    asin_elements = soup.find_all('div', {'data-asin': True})
    asin_list = [i['data-asin'] for i in asin_elements]

contents = {"user-agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0', 'accept-language':'en-IN,en;q=0.9'}
url = "https://www.amazon.in/dp/B0C45N5VPT"

response = requests.get(url,headers=contents)
soup = BeautifulSoup(response.text,'lxml')
title_element = soup.select_one('#productTitle')
print(title_element.contents[0])
# review_element = soup.select("div.review")
# scrapedreviews = []
# for review in review_element:
#     content = review.select_one("span.review-text")
#     scrapedreviews.append(content.text)
# file=open('Scrape.txt','w')
# for i in scrapedreviews:
#     try:
#         if i[-9:] == "Read more":
#             print(True)
#             file.write(i[:-10])
#         else:
#             file.write(i[:-1])
#     except UnicodeEncodeError:
#         pass
