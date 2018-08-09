from __future__ import division
import re
import sys
import random
import requests
import json
import textblob
import pandas as pd	 
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing.pool import Pool
from datetime import datetime
from bs4 import BeautifulSoup
from IPython.display import display
from nltk.tokenize import WordPunctTokenizer
from coala_utils.decorators import generate_ordering

tok = WordPunctTokenizer()


HEADERS_LIST = [
	'Mozilla/5.0 (Windows; U; Windows NT 6.1; x64; fr; rv:1.9.2.13) Gecko/20101203 Firebird/3.6.13',
	'Mozilla/5.0 (compatible, MSIE 11, Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko',
	'Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201',
	'Opera/9.80 (X11; Linux i686; Ubuntu/14.10) Presto/2.12.388 Version/12.16',
	'Mozilla/5.0 (Windows NT 5.2; RW; rv:7.0a1) Gecko/20091211 SeaMonkey/9.23a1pre'
]

HEADER = {'User-Agent': random.choice(HEADERS_LIST)}

INIT_URL_USER = 'https://twitter.com/{u}'

RELOAD_URL_USER = 'https://twitter.com/i/profiles/show/{u}/timeline/tweets?' \
				  'include_available_features=1&include_entities=1&' \
				  'max_position={pos}&reset_error_state=false'


@generate_ordering('timestamp', 'uid', 'text', 'user', 'replies', 'retweets', 'likes')
class Tweet:
	def __init__(self, user, fullname, uid, url, timestamp, text, replies, retweets, likes, html):
		self.user = user.strip('\@')
		self.fullname = fullname
		self.uid = uid
		self.url = url
		self.timestamp = timestamp
		self.text = text
		self.replies = replies
		self.retweets = retweets
		self.likes = likes
		self.html = html

	@classmethod
	def from_soup(cls, tweet):
		return cls(
			user	   = tweet.find('span', 'username').text 													or '',
			fullname   = tweet.find('strong', 'fullname').text 													or '', 
			uid		= tweet['data-item-id'] 														or '',
			url		= tweet.find('div', 'tweet')['data-permalink-path'] 											or '',
			text	   = tweet.find('p', 'tweet-text').text 													or '',
			timestamp  = datetime.utcfromtimestamp(int(tweet.find('span', '_timestamp')['data-time'])),

			replies	= tweet.find('span', 'ProfileTweet-action--reply u-hiddenVisually').find('span', 'ProfileTweet-actionCount')['data-tweet-stat-count'] 		or '0',
			retweets   = tweet.find('span', 'ProfileTweet-action--retweet u-hiddenVisually').find('span', 'ProfileTweet-actionCount')['data-tweet-stat-count'] 	or '0',
			likes	  = tweet.find('span', 'ProfileTweet-action--favorite u-hiddenVisually').find('span', 'ProfileTweet-actionCount')['data-tweet-stat-count'] 	or '0',

			html = str(tweet.find('p', 'tweet-text')) 														or '',
		)
	

	@classmethod
	def from_html(cls, html):
		soup = BeautifulSoup(html, "lxml")
		tweets = soup.find_all('li', 'js-stream-item')
		if tweets:
			for tweet in tweets:
				try:
					yield cls.from_soup(tweet)
				except AttributeError:
					pass 


def query_single_page(url, html_response=True, retry=10, from_user=False):
	try:
		response = requests.get(url, headers=HEADER)
		if html_response:
			html = response.text or ''
		else:
			html = ''
			try:
				json_resp = json.loads(response.text)
				html = json_resp['items_html'] or ''
			except ValueError: pass

		tweets = list(Tweet.from_html(html))

		if not tweets: return [], None
		if not html_response: return tweets, json_resp['min_position']
		return tweets, tweets[-1].uid

	except: pass

	if retry > 0:
		return query_single_page(url, html_response, retry-1)

	return [], None


def query_tweets_from_user(user, limit=None):
	pos = None
	tweets = []
	try:
		while True:
		   new_tweets, pos = query_single_page(INIT_URL_USER.format(u=user) if pos is None else RELOAD_URL_USER.format(u=user, pos=pos), pos is None, from_user=True)
		   if len(new_tweets) == 0:
			   return tweets

		   tweets += new_tweets
		   if limit and len(tweets) >= limit:
			   return tweets
	
	except Exception:
		print("Unexpected Error")

	return tweets


def data_preparation(tweet): #TO DO: stemming, lemmatization

	url_regex = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'

	clean = re.sub(url_regex, '', tweet, flags = re.MULTILINE)                                                # strip out urls. urls, ew, nasty.
	clean = clean.replace('\n', ' ').replace("'", " ").replace('"', ' ')

	try:	
		clean = clean.decode("utf-8-sig").replace(u"\ufffd", "?")                                         # strip out Byte Order Marks
	except:
		pass

	clean = re.sub(r'[^a-zA-Z ]', '', clean, flags = re.MULTILINE)                                            # the "#" symbol is actually called octothorpe. bananas.
	clean = (" ".join(tok.tokenize(clean))).strip()                                                           # Tokenization

	return clean


def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = textblob.TextBlob(data_preparation(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


def main():

	tweets = query_tweets_from_user(sys.argv[1], limit=50)

	data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

	data['len']  = np.array([len(tweet.text) for tweet in tweets])
	data['ID']   = np.array([tweet.uid for tweet in tweets])
	data['Date'] = np.array([tweet.timestamp for tweet in tweets])
	data['Likes']  = np.array([tweet.likes for tweet in tweets])
	data['RTs']	= np.array([tweet.retweets for tweet in tweets])

	for tweet in data['Tweets'].head(1):
		print("\nORIGINAL: {}".format(tweet))
		print("\nSANITIZED: {}\n\n".format(data_preparation(tweet)))
	
	
	data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])
	display(data.head(100))

	pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
	neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
	neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]

	print("\n\nPercentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
	print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
	print("Percentage de negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))


if __name__ == "__main__":
	main()


"""
username='POTUS'
url = 'https://www.twitter.com/' + username
r = requests.get(url)
soup = BeautifulSoup(r.content, "lxml")

f = soup.find('li', class_="ProfileNav-item--followers").find('a')['title'] or ""
num_followers = int(f.split(' ')[0].replace(',','').replace('.', ''))


location = " ".join((soup.find('span', {'class': 'ProfileHeaderCard-locationText u-dir'}).text.replace('\n', '') or "").split())

print(location)
print(num_followers)
"""


"""
conn = sqlite3.connect('twitter_testing.sqlite')
cur = conn.cursor()
cur.executescript('''

CREATE TABLE Tweets_London (
    id     INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    user_id TEXT,
    user_name TEXT,
    user_timezone TEXT,
    user_language TEXT,
    detected_language TEXT,
    tweet_text  TEXT,
    tweet_created TEXT
)
''')


