#code pulled heavily from: See LICENSE

from __future__ import division
import random
import requests
import json
import textblob
import datetime as dt
from datetime import datetime
from bs4 import BeautifulSoup
from coala_utils.decorators import generate_ordering


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
		pass

	return tweets


def get_data(screenname, limit):
	
	tweets = query_tweets_from_user(screenname, limit=limit)

	return tweets





