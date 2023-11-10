import requests
from bs4 import BeautifulSoup

contents = {"user-agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0', 'accept-language':'en-IN,en;q=0.9'}
url = "https://www.amazon.in/dp/B0C45N5VPT?pd_rd_w=a1QS5&content-id=amzn1.sym.e9ba7ce7-f932-416d-a507-56007d975bc6&pf_rd_p=e9ba7ce7-f932-416d-a507-56007d975bc6&pf_rd_r=G3837PSV5A4KAS6XVCYR&pd_rd_wg=4V1Xo&pd_rd_r=8c10399d-fc91-430d-97b7-3348149129b2"

response = requests.get(url,headers=contents)
soup = BeautifulSoup(response.text,'lxml')
title_element = soup.select_one('#productTitle')
print(title_element)
review_element = soup.select("div.review")
scrapedreviews = []
for review in review_element:
    content = review.select_one("span.review-text")
    scrapedreviews.append(content.text)
file=open('Scrape.txt','w')
for i in scrapedreviews:
    if i[-9:] == "Read more":
        print(True)
        file.write(i[:-10])
    else:
        file.write(i[:-1])

