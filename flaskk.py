from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def hello():
    message = "Welcome to the Flask example!"
    return render_template('search_results.html', message=message)

@app.route('/process_url', methods=['POST'])
def process_url():
    product_url = request.form.get('product_url')
    return f"You entered the following URL: {product_url}"

if __name__ == '__main__':
    app.run(debug=True)
