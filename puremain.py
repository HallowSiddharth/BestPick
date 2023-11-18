import requests
from bs4 import BeautifulSoup
import time
import random
import emoji
import analyzer
from flask import Flask, request, render_template

app = Flask(__name__)

def search(searchterm):
    words = searchterm.split()
    query = "+".join(words)
    request = f"https://www.amazon.in/s?k={query}"
    contents = {"user-agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0', 'accept-language':'en-IN,en;q=0.9'}
    response = requests.get(request,headers=contents)
    soup = BeautifulSoup(response.text,'lxml')
    asin_elements = soup.find_all('div', {'data-asin': True})
    asin_list = [i['data-asin'] for i in asin_elements if i['data-asin'] !='']
    return asin_list

def extract_from_asin(asin):
    contents = {"user-agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0', 'accept-language':'en-IN,en;q=0.9'}

    time.sleep(random.uniform(1.0,2.0))
    url = f"https://www.amazon.in/dp/{asin}"

    response = requests.get(url,headers=contents)
    soup = BeautifulSoup(response.text,'lxml')
    title_element = soup.select_one('#productTitle')
    #print(title_element)
    review_element = soup.select("div.review")
    scrapedreviews = []
    for review in review_element:
        content = review.select_one("span.review-text")
        if content is not None:
            scrapedreviews.append(content.text)
    final = []
    for review in scrapedreviews:
        rev = ''.join(c for c in review if c not in emoji.EMOJI_DATA)
        final.append(rev)
    return title_element.contents[0],final



def process_query(query):
    asins = search(query)
    time.sleep(random.uniform(0.2,1.5))
    master_reviews = {}
    for asin in asins:
        name,reviews = extract_from_asin(asin)
        master_reviews[(asin,name)] = reviews
    return master_reviews

def generate_score(reviews):
    pass

@app.route('/')
def hello():
    message = "Welcome to the Flask example!"
    return render_template('index.html', message=message)

# @app.route('/process_url', methods=['POST'])
# def process_url():
#     product_url = request.form.get('product_url')
#     return f"You entered the following URL: {product_url}"

# @app.route('/process_url', methods=['POST'])
# def search_endpoint():
#     query = request.form.get('product_url')
#     d = process_query(query)
#     answers = []
#     for product in d:
#         asin, title = product
#         reviews = d[product]
#         score = analyzer.train_with_data(reviews, "dataset2.csv")
#         image_url = get_image_url_for_product(asin)  
#         answers.append([score, title, image_url])
#     answers.sort(reverse=True)  # Sort results by score in descending order

#     best = answers[0] if answers else None
#     second = answers[1] if len(answers) > 1 else None
#     third = answers[2] if len(answers) > 2 else None
#     print('ANSWERS')
#     print(answers)
#     print(best,second,third)
#     print(best[0],best[1],best[2])

#     return render_template('search_results.html', best=best, second=second, third=third)
def get_product_and_image_urls(asin):
    contents = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "accept-language": "en-IN,en;q=0.9"
}

    
    # Construct the URL for the product page using the ASIN
    product_url = f"https://www.amazon.in/dp/{asin}"

    # Send a request to the product page
    response = requests.get(product_url, headers=contents)
    soup = BeautifulSoup(response.text, 'lxml')

    # Extract the product image URL from the page (you may need to adjust the selector)
    image_element = soup.select_one('#landingImage')  # Adjust the selector based on the HTML structure of the page

    # If the image element is found, retrieve the image URL
    image_url = image_element.get('src') if image_element else None

    # Return both the product and image URLs
    print(product_url,image_url)
    return product_url, image_url

@app.route('/process_url', methods=['POST'])
def search_endpoint():
    query = request.form.get('product_url')
    d = process_query(query)
    answers = []
    for product in d:
        asin, title = product
        reviews = d[product]
        score = analyzer.train_with_data(reviews, "dataset2.csv")
        product_url, image_url = get_product_and_image_urls(asin)
        answers.append([score, title, product_url, image_url])
    answers.sort(reverse=True)  # Sort results by score in descending order

    best = answers[0] if answers else None
    second = answers[1] if len(answers) > 1 else None
    third = answers[2] if len(answers) > 2 else None
    print('ANSWERS')
    print(answers)
    print(best, second, third)
    print(best[0], best[1], best[2], best[3])

    return render_template('search_results.html', best=best, second=second, third=third)

@app.route('/get_image_url_for_product', methods=['POST'])
def get_image_url_for_product(asin):
    # Assuming the ASIN is valid
    product_url = f"https://www.amazon.in/dp/{asin}"
    return product_url

if __name__ == '__main__':
    app.run(debug=True)
