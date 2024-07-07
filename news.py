import requests
from bs4 import BeautifulSoup
from newspaper import Article
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

def fetch_news(keyword, base_url="https://news.google.com", num_articles=20):
    """Fetch news URLs containing the specified keyword from Google News."""
    search_url = f"{base_url}/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all news articles
    articles = soup.find_all('a', href=True)
    news_links = []
    
    for a in articles:
        href = a['href']
        if href.startswith('./articles'):
            full_link = base_url + href[1:]
            news_links.append(full_link)
        if len(news_links) == num_articles:  # Stop collecting links after reaching the desired number
            break
    
    return news_links  # No need to remove duplicates since we stop at a fixed number

def analyze_sentiment(news_links):
    """Perform sentiment analysis on the content of each news link."""
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    for index, url in enumerate(news_links):
        try:
            print(f"Processing article number: {index + 1}/{len(news_links)}")
            article = Article(url)
            article.download()
            article.parse()
            content = article.text
            if content:
                score = sia.polarity_scores(content)
                sentiment_scores.append(score['compound'])  # Use compound score
        except Exception as e:
            print(f"Failed to process {url}: {e}")
    
    if sentiment_scores:
        positive = sum(1 for score in sentiment_scores if score > 0)
        negative = sum(1 for score in sentiment_scores if score < 0)
        neutral = len(sentiment_scores) - positive - negative
        
        positive_percent = 100 * positive / len(sentiment_scores)
        negative_percent = 100 * negative / len(sentiment_scores)
        neutral_percent = 100 * neutral / len(sentiment_scores)
        
        return [positive_percent, negative_percent, neutral_percent]

def getNews(keyword, num_articles=30):
    news_links = fetch_news(keyword, num_articles=num_articles)
    return analyze_sentiment(news_links)