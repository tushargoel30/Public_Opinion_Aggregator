# Flask endpoint that sends data
from flask import Flask, render_template, request, redirect, url_for, session,jsonify
from lookup import callup
from news import getNews
from trend import getPrediction
from keygen import genCloud
from region import regional
from process import processSearch

app = Flask(__name__)
app.secret_key = '123'  # Needed for session management

# @app.route('/get-data')
# def get_data():
#     # Prepare your data, e.g., a list of x and y coordinates
#     data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
#     return jsonify(data)

@app.route('/')
def home():
    return render_template('new.html')

@app.route('/search', methods=['POST'])
def search():
    print('searching called!!!!!!')
    user_input = request.form['search_input']
    user_input=processSearch(user_input)  # Process the user input
    data=callup(user_input)  # Get results from the callup function
    genCloud(data)
    regional(user_input)
    return jsonify({'status': 'success'})  # Notify the front-end about the success

@app.route('/news_Search', methods=['POST'])
def search_news():
    # Placeholder for real news data fetching
    user_input = request.form['search_input']
    results = getNews(user_input)  # Get results from the callup function
    session['news'] = results
    return jsonify({'status': 'success'})

@app.route('/trend_Search', methods=['POST'])
def search_trends():
    # Placeholder for real news data fetching
    user_input = request.form['search_input']
    getPrediction(user_input)  # Get results from the callup function
    return jsonify({'status': 'success'})


# @app.route('/sentiment-data')
# def sentiment_data():
#     # Placeholder for real sentiment data fetching
#     reddata = session.get('reddit', 'No results found.')
#     print(reddata)
#     return reddata

@app.route('/sentiment-data')
def sentiment_data():
    print('sentiment data called!!!!!!')
    try:
        return app.send_static_file('static/results.json')
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/trend-data')
def trend_data():
    # Placeholder for real news data fetching
    data = session.get('trends', 'No trends found.')
    return jsonify(result=data)



@app.route('/news-data')
def news_data():
    # Placeholder for real news data fetching
    data = session.get('news', 'No news found.')
    return data

@app.route('/results')
def results():
    # Retrieve results from session
    # results = session.get('result', 'No results found.')
    return render_template('results.html')  # Pass results to the template


@app.route('/profile')
def profile():
    return render_template('users-profile.html')

@app.route('/faq')
def faq():
    return render_template('pages-faq.html')

@app.route('/contact')
def contact():
    return render_template('pages-contact.html')

@app.route('/register')
def register():
    return render_template('pages-register.html')

@app.route('/login')
def login():
    return render_template('pages-login.html')


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')