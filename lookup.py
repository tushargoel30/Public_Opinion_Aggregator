import praw
import pandas as pd
# Import Basic Libraries
import re
import os
import pandas as pd
import numpy as np
from datetime import datetime

from better_profanity import profanity
from textblob import TextBlob

from prawcore.exceptions import Forbidden, PrawcoreException
# Import NLP Libraries
import nltk
from nltk.corpus import stopwords
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# # downloading stopwords
# nltk.download('stopwords')

# Remove distarcting warning
import warnings
warnings.filterwarnings('ignore')

import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# Define the stopwords list
stop_words = set(stopwords.words('english'))

def tweet_preprocessing(tweet):
    # Convert to lower case
    tweet = tweet.lower()
    # Remove HTML code using Beautiful Soup
    tweet = BeautifulSoup(tweet, "html.parser").get_text()
    # Remove URLs using regular expressions
    tweet = re.sub(r"http\S+", "", tweet)
    # Censor profanity
    # profanity.load_censor_words()
    # tweet = profanity.censor(tweet)
    # Remove Twitter handles
    tweet = re.sub('@[^\s]+', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'\B#\S+', '', tweet)
    # Remove special characters and punctuations
    tweet = re.sub(r'\W', ' ', tweet)
    # Remove single characters except for 'a' and 'i'
    tweet = re.sub(r'\s+[a-hj-z]\s+', ' ', tweet)
    tweet = re.sub(r'\s+i\s+', ' I ', tweet)
    tweet = re.sub(r'\s+a\s+', ' a ', tweet)
    # Remove numbers
    tweet = re.sub(r'\d+', '', tweet)
    # Substitute multiple spaces with single space
    tweet = re.sub(r'\s+', ' ', tweet)
    # Remove stop words
    tweet = ' '.join([word for word in tweet.split() if word not in stop_words])
    return tweet


def get_reddit_posts(keyword, num_posts=100):
    client_id = "mg08wSP1NEHdCxactwgojA"
    client_secret = "xpXS3RiFs_tQk4f43vFDo_eLBbLYjA"
    user_agent = "myAPI/0.0.1"

    # Authenticate with the Reddit API
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    except PrawcoreException as e:
        print(f"Error connecting to Reddit API: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on connection failure

    data = []
    subreddits = ["india", "politics", "IndiaSpeaks", "librandu", 
                  "neoliberal", "unitedstatesofindia", "IndianModerate", "worldpolitics","indiadiscussion","IndianHistory"]

    # Loop through the list of subreddits
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            # Use pagination to fetch more results
            for submission in subreddit.search(keyword, limit=num_posts, sort='new'):
                title = submission.title
                content = submission.selftext
                url = submission.url

                # Ignore posts with empty content
                if content:
                    data.append({'Title': title, 'Content': content, 'URL': url})

                # Break the loop if the desired number of posts is reached
                if len(data) >= num_posts:
                    break

        except Forbidden:
            print(f"Access to subreddit {subreddit_name} is forbidden. Skipping...")
        except PrawcoreException as e:
            print(f"Error fetching data from subreddit {subreddit_name}: {e}")
        # Optionally, break the loop if we have enough posts across all subreddits
        if len(data) >= num_posts:
            break

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    return df

def analyze(df):
    # Preprocess the content
    df['Content'] = df['Content'].apply(tweet_preprocessing)

    # Calculate the sentiment of each post
    sentiment_values = [[TextBlob(content).sentiment.polarity, content] for content in df['Content']]
    sentiment = []

    for i in range(len(sentiment_values)):
        a = sentiment_values[i][0]
        b = sentiment_values[i][1]
        if a > 0:
            sentiment.append(["Positive", b])
        elif a < 0:
            sentiment.append(["Negative", b])
        else:
            sentiment.append(["Neutral", b])

    sentiment_df = pd.DataFrame(sentiment, columns=["Sentiment", "Tweet"])

    # Count the number of posts with each sentiment
    sentiment_count = sentiment_df['Sentiment'].value_counts()

    # Calculate the percentage of each sentiment
    positive_percent = sentiment_count['Positive'] / len(sentiment_df) * 100
    negative_percent = sentiment_count['Negative'] / len(sentiment_df) * 100
    neutral_percent = sentiment_count['Neutral'] / len(sentiment_df) * 100

    return [positive_percent, negative_percent, neutral_percent]


def extract_corpus_keywords_with_nmf(result_df, max_features=500, n_topics=5, top_n=10):
    """
    Extract most significant keywords from the entire corpus using NMF for topic modeling,
    with additional checks to remove duplicates and improve relevance, and provide scores.

    Parameters:
    result_df (DataFrame): DataFrame containing text data in 'Content' column.
    max_features (int): The maximum number of features for the TF-IDF vectorizer.
    n_topics (int): The number of topics to identify in the corpus.
    top_n (int): The number of top keywords to return from the entire corpus.

    Returns:
    dict: A dictionary containing the top N keywords and their respective relevance scores.
    """
    texts = result_df['Content'].tolist()
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))

    # Fit and transform the texts to a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Apply NMF for topic modeling
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(tfidf_matrix)  # Only fit to get the components matrix H

    # Aggregating word weights across all topics
    aggregated_weights = np.sum(nmf.components_, axis=0)
    top_indices = np.argsort(aggregated_weights)[-top_n:][::-1]
    top_keywords = [(feature_names[i], aggregated_weights[i]) for i in top_indices]

    # Post-processing to remove duplicates and improve relevance
    keywords_with_scores = {}
    seen = set()
    for keyword, score in top_keywords:
        # Split keyword into individual words to check for redundancy
        words = keyword.split()
        if len(words) > 1 and any(word in seen for word in words):
            continue  # skip n-grams that are redundant with existing keywords
        seen.update(words)
        keywords_with_scores[keyword] = score
        if len(keywords_with_scores) == top_n:
            break

    return keywords_with_scores




# [57.99999999999999, 23.0, 19.0]
def callup(keyword_to_search):
    result_df = get_reddit_posts(keyword_to_search, num_posts=100)  
    result = analyze(result_df)
    print(keyword_to_search)
    print(result)
    # Display the DataFrame
    with open('static/results.json', 'w') as f:
        json.dump(result, f)
    corpus_keywords = extract_corpus_keywords_with_nmf(result_df, max_features=500, n_topics=len(result_df), top_n=30)
    # genCloud(corpus_keywords)
    return corpus_keywords

