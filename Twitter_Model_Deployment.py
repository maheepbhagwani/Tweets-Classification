import tweepy
import re, nltk
import time
import numpy as np
import pandas as pd
import csv
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

def twitter_api():
    CONSUMER_KEY = "Your Key Here"
    CONSUMER_SECRET = "Your Secret Here"
    ACCESS_TOKEN = "Your Token Here"
    ACCESS_TOKEN_SECRET = "Your Token Secret Here"
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    return tweepy.API(auth)

def twitter_search(api, query, max_tweets):
    tweets = []
    initial_tweets = api.search(q=query, lang='en', count=100, tweet_mode = "extended", truncated = False)
    tweets.extend(initial_tweets)
    max_id = initial_tweets[-1].id
    tweet_counter = 0
    while len(tweets) < max_tweets:
        new_tweets = api.search(q=query, lang='en', count=100, tweet_mode = "extended", truncated = False, max_id=str(max_id-1))
        if not new_tweets:
            break
        else:
            tweets.extend(new_tweets)
            max_id = new_tweets[-1].id
            tweet_counter += len(new_tweets)
            if tweet_counter > 15000:
                print("sleeping for 16 minutes")
                time.sleep(16*60)
                tweet_counter = 0

    print("Number of tweets retrieved: ", len(tweets))
    return tweets

def create_csv(query, outtweets):
    with open('%s_tweets.csv' % query, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","tweets"])
        writer.writerows(outtweets)

def normalizer(tweet):
    soup = BeautifulSoup(tweet, 'lxml')   # removing HTML encoding such as ‘&amp’,’&quot’
    souped = soup.get_text()
    only_words = re.sub("(@[A-Za-z0-9]+)|([^A-Za-z \t])|(\w+:\/\/\S+)"," ", souped) # removing @mentions, hashtags, urls

    tokens = nltk.word_tokenize(only_words)
    removed_letters = [word for word in tokens if len(word)>2]
    lower_case = [l.lower() for l in removed_letters]

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

def main():
    #### Loading the saved model
    model = joblib.load('svc.sav')
    vocabulary_model = pd.read_csv('vocabulary_SVC.csv', header=None)
    vocabulary_model_dict = {}
    for i, word in enumerate(vocabulary_model[0]):
         vocabulary_model_dict[word] = i
    tfidf = TfidfVectorizer(sublinear_tf=True, vocabulary = vocabulary_model_dict, min_df=5, norm='l2', ngram_range=(1,3)) # min_df=5 is clever way of feature engineering
    #### Retrieving tweets for user query
    api = twitter_api()
    query = input("Please enter your query: ")
    tweet_query = query+" -filter:retweets"
    max_tweets = 5000 # maximum number of tweets to be retrieved
    retrieved_tweets = twitter_search(api, tweet_query, max_tweets)
    outtweets = [[tweet.id_str, tweet.created_at, tweet.full_text.encode("utf-8")] for tweet in retrieved_tweets]
    create_csv(query, outtweets)
    #### Reading retrieved tweets as dataframe
    tweet_df = pd.read_csv('%s_tweets.csv' % query, encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
    #### Normalizing retrieved tweets
    tweet_df['normalized_tweets'] = tweet_df.tweets.apply(normalizer)
    tweet_df = tweet_df[tweet_df['normalized_tweets'].map(len) > 0] # removing rows with normalized tweets of length 0
    print("Number of tweets remaining after cleaning: ", tweet_df.normalized_tweets.shape[0])
    print(tweet_df[['tweets','normalized_tweets']].head())
    #### Saving cleaned tweets to csv file
    tweet_df.drop(['id', 'created_at'], axis=1, inplace=True)
    tweet_df.to_csv('cleaned_tweets.csv', encoding='utf-8', index=False)
    cleaned_tweets = pd.read_csv("cleaned_tweets.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1)
    cleaned_tweets_tfidf = tfidf.fit_transform(cleaned_tweets['normalized_tweets'])
    targets_pred = model.predict(cleaned_tweets_tfidf)
    #### Saving predicted sentiment of tweets to csv
    cleaned_tweets['predicted_sentiment'] = targets_pred.reshape(-1,1)
    cleaned_tweets.to_csv('predicted_sentiment.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    main()
