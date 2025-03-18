import tweepy
import requests
import json
import schedule
import time
import os
import logging
from datetime import datetime
from pymongo import MongoClient


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from environment variables
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET_KEY = os.getenv('TWITTER_API_SECRET_KEY')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

MONGO_URI = os.getenv('MONGO_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME')

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
posts_collection = db['posts']


class TwitterBot:
    def __init__(self):
        auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET_KEY)
        auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
        self.api = tweepy.API(auth)

    def post_tweet(self, message):
        logger.info(f"Posting tweet: {message}")
        try:
            self.api.update_status(message)
            logger.info("Tweet posted successfully.")
        except tweepy.TweepError as e:
            logger.error(f"Error posting tweet: {e}")

    def fetch_tweets(self, username):
        logger.info(f"Fetching tweets from: {username}")
        tweets = self.api.user_timeline(screen_name=username, count=10, tweet_mode="extended")
        return [{'text': tweet.full_text, 'created_at': tweet.created_at} for tweet in tweets]


class SocialMediaScheduler:
    def __init__(self):
        self.twitter_bot = TwitterBot()

    def schedule_post(self, message, post_time):
        logger.info(f"Scheduling post: '{message}' at {post_time}")
        schedule.every().day.at(post_time).do(self.twitter_bot.post_tweet, message)

    def run(self):
        while True:
            schedule.run_pending()
            time.sleep(1)


class PostManager:
    def __init__(self):
        self.posts = []

    def load_posts(self):
        logger.info("Loading posts from database...")
        self.posts = list(posts_collection.find({}))
        logger.info(f"Loaded {len(self.posts)} posts.")

    def add_post(self, message):
        logger.info(f"Adding post to the database: {message}")
        post_data = {
            'message': message,
            'created_at': datetime.now()
        }
        posts_collection.insert_one(post_data)
        self.load_posts()


def main():
    logging.info("Starting Social Media Automation...")
    
    post_manager = PostManager()
    post_manager.load_posts()

    scheduler = SocialMediaScheduler()
    
    # Example hardcoded posts for demonstration purposes
    posts_to_schedule = [
        {"message": "Hello World! #myfirsttweet", "time": "09:00"},
        {"message": "Automating tweets is fun! #TwitterBot", "time": "12:00"},
        {"message": "Stay tuned for more updates!", "time": "15:00"},
    ]

    for post in posts_to_schedule:
        post_manager.add_post(post['message'])
        scheduler.schedule_post(post['message'], post['time'])

    scheduler.run()


if __name__ == "__main__":
    main()