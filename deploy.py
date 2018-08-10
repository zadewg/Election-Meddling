from __future__ import division
import re
import sys
import random
import argparse
import requests
import json
import textblob
import nltk
import pandas as pd	 
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from functools import partial, reduce
from multiprocessing.pool import Pool
from datetime import datetime
from bs4 import BeautifulSoup
from IPython.display import display
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from coala_utils.decorators import generate_ordering


def safecall(function, default=0, exception=Exception, *args):
	try:
		return function(*args)
	except exception:
		return default


def banner():

	print("       ___           ___                       ___      ")
	print("      /\  \         /\  \          ___        /\__\     ")
	print("     /::\  \       /::\  \        /\  \      /::|  |    ")
	print("    /:/\ \  \     /:/\:\  \       \:\  \    /:|:|  |    ")
	print("   _\:\~\ \  \   /::\~\:\  \      /::\__\  /:/|:|  |__  ")
	print("  /\ \:\ \ \__\ /:/\:\ \:\__\  __/:/\/__/ /:/ |:| /\__\ ")
	print("  \:\ \:\ \/__/ \/__\:\/:/  / /\/:/  /    \/__|:|/:/  / ")
	print("   \:\ \:\__\        \::/  /  \::/__/         |:/:/  /  ")
	print("    \:\/:/  /         \/__/    \:\__\         |::/  /   ")
	print("     \::/  /                    \/__/         /:/  /    ")
	print("      \/__/                                   \/__/     ")

	print("\n                  Red Team Intelligence               ")

	print("\n     https://github.com/zadewg/Election-Meddling  \n\n")


def parsein():

	global TARGETS, NUM, BENEFIT

	parser = argparse.ArgumentParser(description='https://github.com/zadewg/Election-Meddling')
	parser.add_argument('-t','--target', help='Target sername(s)', required=True)
	parser.add_argument('-c','--count', help='Number of tweets to retrieve from user.', required=True)
	parser.add_argument('-b','--benefit', help='Political wing to benefit', required=True)
	args = vars(parser.parse_args())

	TARGETS = [args['target']]
	NUM = args['count']
	BENEFIT = args['benefit']


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


def query_tweets_from_user(user, limit=99):
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



class Splitter(object):
    """
    split the document into sentences and tokenize each sentence
    """
    def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self,text):
        """
        out : ['What', 'can', 'I', 'say', 'about', 'this', 'place', '.']
        """
        # split into single sentence
        sentences = self.splitter.tokenize(text)
        # tokenization in each sentences
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        return tokens


class LemmatizationWithPOSTagger(object):
    def __init__(self):
        pass
    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def pos_tag(self,tokens):

        pos_tokens = [nltk.pos_tag(token) for token in tokens]
        pos_tokens = [ [(word, lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)), [pos_tag]) for (word,pos_tag) in pos] for pos in pos_tokens]
        return pos_tokens



lemmatizer = WordNetLemmatizer()
splitter = Splitter()
lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()

def data_preparation(tweet): #nltk.tag._POS_TAGGER #treebank tag set https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
	
	url_regex = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'

	clean = re.sub(url_regex, '', tweet, flags = re.MULTILINE)                                                # strip out urls. urls, ew, nasty.
	clean = clean.replace('\n', ' ').replace("'", " ").replace('"', ' ')

	try:	
		clean = clean.decode("utf-8-sig").replace(u"\ufffd", "?")                                         # strip out Byte Order Marks
		print("Detected BOS")
	except:
		pass
	
	clean = re.sub(r'[^a-zA-Z ]', '', clean, flags = re.MULTILINE)                                            # the "#" symbol is actually called octothorpe. bananas.
	
	tokens = splitter.split(clean)										  # Tokeniztion

	lemma_pos_token = lemmatization_using_pos_tagger.pos_tag(tokens)					  # Part of speech tagging.
	out = ' '.join([out[1] for out in lemma_pos_token[0]])
	return out

	''' #https://pypi.org/project/hunspell/ #Double tokenizing. hunspell for units, nltk for context.
	import hunspell

	hobj = hunspell.HunSpell('/usr/share/myspell/en_US.dic', '/usr/share/myspell/en_US.aff')
	hobj.spell('spookie')

	hobj.suggest('spookie')

	hobj.spell('spooky')

	hobj.analyze('linked')

	hobj.stem('linked')
	'''


def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''

    try:
       analysis = textblob.TextBlob(data_preparation(tweet))
    except:
       analysis = textblob.TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1



def main(screenname, num):

	tweets = query_tweets_from_user(screenname, limit=num)

	data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

	data['len']  = np.array([len(tweet.text) for tweet in tweets])
	data['ID']   = np.array([tweet.uid for tweet in tweets])
	data['Date'] = np.array([tweet.timestamp for tweet in tweets])
	data['Likes']  = np.array([tweet.likes for tweet in tweets])
	data['RTs']	= np.array([tweet.retweets for tweet in tweets])
	
	data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])
	display(data.head(100))

	pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
	neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
	neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]

	print("\n\nPercentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
	print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
	print("Percentage de negative tweets: {}%\n\n\n".format(len(neg_tweets)*100/len(data['Tweets'])))



#	for tweet in data['Tweets'].head(1):
#		print("\n\nORIGINAL SAMPLE: {}".format(tweet))
#		print("\nSANITIZED SAMPLE: {}\n\n".format(data_preparation(tweet)))
		
		
	hillarySentiments = [0]
	hillaryKeywords = ['hillary', 'clinton', 'hillaryclinton']

	trumpSentiments = [0]
	trumpKeywords = ['trump', 'realdonaldtrump']

	cruzSentiments = [0]
	cruzKeywords = ['cruz', 'tedcruz']

	bernieSentiments = [0]
	bernieKeywords = ['bern', 'bernie', 'sanders', 'sensanders']

	obamaSentiments = [0]
	obamaKeywords = ['obama', 'barack', 'barackobama']

	republicanSentiments = [0]
	republicanKeywords = ['republican', 'conservative']

	democratSentiments = [0]
	democratKeywords = ['democrat', 'dems', 'liberal']
	
	gunsSentiments = [0]
	gunsKeywords = ['guns', 'gun', 'nra', 'pistol', 'firearm', 'shooting', 'ar15']

	immigrationSentiments = [0]
	immigrationKeywords = ['immigration', 'immigrants', 'citizenship', 'naturalization', 'visas']

	employmentSentiments = [0]
	emplyomentKeywords = ['jobs', 'employment', 'unemployment', 'job']

	inflationSentiments = [0]
	inflationKeywords = ['inflate', 'inflation', 'price hike', 'price increase', 'prices rais']

	minimumwageupSentiments = [0]
	minimumwageupKeywords = ['raise minimum wage', 'wage increase', 'raise wage', 'wage hike']

	abortionSentiments = [0]
	abortionKeywords = ['abortion', 'pro-choice', 'planned parenthood']

	governmentspendingSentiments = [0]
	governmentspendingKeywords = ['gov spending', 'government spending', 'gov. spending', 'expenditure']

	taxesupSentiments = [0]
	taxesupKeywords = ['raise tax', 'tax hike', 'taxes up', 'tax up', 'increase taxes', 'taxes increase', 'tax increase']

	taxesdownSentiments = [0]
	taxesdownKeywords = ['lower tax', 'tax cut', 'tax slash', 'taxes down', 'tax down', 'decrease taxes', 'taxes decrease', 'tax decrease']
	
	deathpenaltySentiments = [0]
	deathpenaltyKeywords = ['aaaaaaaaaaaaaaaaaaaaa']

	healthcareSentiments = [0]
	healthcareKeywords = ['aaaaaaaaaaaaaaaaaaaaa']
	
	lgbtSentiments = [0]
	lgbtKeywords = ['aaaaaaaaaaaaaaaaaaaaa']
	
	environmentSentiments = [0]
	environmentKeywords = ['aaaaaaaaaaaaaaaaaaaaa']

	welfareSentiments = [0]
	welfareKeywords = ['aaaaaaaaaaaaaaaaaaaaa']

	socialSentiments = [0]
	socialKeywords = ['aaaaaaaaaaaaaaaaaaaaa']

	laborunionSentiments = [0]
	laborunionKeywords = ['aaaaaaaaaaaaaaaaaaaaa']

	drugsSentiments = [0]
	drugsKeywords = ['aaaaaaaaaaaaaaaaaaaaa']

	warSentiments = [0]
	warKeywords = ['aaaaaaaaaaaaaaaaaaaaa']


	#(nameOfTuple, sentimentList, keywordList)
	personSentimentList = [
					('Hillary', hillarySentiments, hillaryKeywords), 
					('Trump', trumpSentiments, trumpKeywords), 
					('Cruz', cruzSentiments, cruzKeywords), 
					('Bernie', bernieSentiments, bernieKeywords), 
					('Obama', obamaSentiments, obamaKeywords)]
	issueSentimentList = [
					('Guns', gunsSentiments, gunsKeywords), 
					('Immigration', immigrationSentiments, immigrationKeywords), 
					('Employment', employmentSentiments, emplyomentKeywords), 
					('Inflation', inflationSentiments, inflationKeywords),
					('Minimum wage up', minimumwageupSentiments, minimumwageupKeywords), 
					('Abortion', abortionSentiments, abortionKeywords),
					('Government spending', governmentspendingSentiments, governmentspendingKeywords), 
					('Taxes up', taxesupSentiments, taxesupKeywords), 
					('Taxes down', taxesdownSentiments, taxesdownKeywords),
					('Death penalty', deathpenaltySentiments, deathpenaltyKeywords),
					('Health care', healthcareSentiments, healthcareKeywords),
					('LGBT', lgbtSentiments, lgbtKeywords),
					('Environment', environmentSentiments, environmentKeywords),
					('Welfare', welfareSentiments, welfareKeywords),
					('Social', socialSentiments, socialKeywords),
					('Labor Union', laborunionSentiments, laborunionKeywords),
					('Drugs', drugsSentiments, drugsKeywords),
					('War', warSentiments, warKeywords),
			     ]
	

	MENTIONS = 0
	
	try:
		for tweet in data['Tweets']:
			for person in personSentimentList:
				for keyword in person[2]:
					if keyword in tweet:
						tb = textblob.TextBlob(data_preparation(tweet))
						person[1].append(tb.sentiment.polarity)
						MENTIONS += 1
						break

			for issue in issueSentimentList:
				for keyword in issue[2]:
					if keyword in tweet:
						tb = textblob.TextBlob(data_preparation(tweet))
						issue[1].append(tb.sentiment.polarity)
						MENTIONS += 1
						
					break

	except:
		print(sys.exc_info()[0])
		sys.stdout.flush()


	print("\nDetected {} mentions.\n".format(MENTIONS))


	
	hillary = 0 if (len(hillarySentiments) == 1) else reduce(lambda x, y: x + y, hillarySentiments) / float(len(hillarySentiments)-1)

	trump = 0 if (len(trumpSentiments) == 1) else reduce(lambda x, y: x + y, trumpSentiments) / float(len(trumpSentiments)-1)

	cruz = 0 if (len(cruzSentiments) == 1) else reduce(lambda x, y: x + y, cruzSentiments) / float(len(cruzSentiments)-1)

	bernie = 0 if (len(bernieSentiments) == 1) else reduce(lambda x, y: x + y, bernieSentiments) / float(len(bernieSentiments)-1)

	obama = 0 if (len(obamaSentiments) == 1) else reduce(lambda x, y: x + y, obamaSentiments) / float(len(obamaSentiments)-1)

	republican = 0 if (len(republicanSentiments) == 1) else reduce(lambda x, y: x + y, republicanSentiments) / float(len(republicanSentiments)-1)

	democrat = 0 if (len(democratSentiments) == 1) else reduce(lambda x, y: x + y, democratSentiments) / float(len(democratSentiments)-1)

	guns = 0 if (len(gunsSentiments) == 1) else reduce(lambda x, y: x + y, gunsSentiments) / float(len(gunsSentiments)-1)

	immigration = 0 if (len(immigrationSentiments) == 1) else reduce(lambda x, y: x + y, immigrationSentiments) / float(len(immigrationSentiments)-1)

	employment = 0 if (len(employmentSentiments) == 1) else reduce(lambda x, y: x + y, employmentSentiments) / float(len(employmentSentiments)-1)

	inflation = 0 if (len(inflationSentiments) == 1) else reduce(lambda x, y: x + y, inflationSentiments) / float(len(inflationSentiments)-1)

	wageup = 0 if (len(minimumwageupSentiments) == 1) else reduce(lambda x, y: x + y, minimumwageupSentiments) / float(len(minimumwageupSentiments)-1)

	abortion = 0 if (len(abortionSentiments) == 1) else reduce(lambda x, y: x + y, abortionSentiments) / float(len(abortionSentiments)-1)

	govspend = 0 if (len(governmentspendingSentiments) == 1) else reduce(lambda x, y: x + y, governmentspendingSentiments) / float(len(governmentspendingSentiments)-1)

	taxup = 0 if (len(taxesupSentiments) == 1) else reduce(lambda x, y: x + y, taxesupSentiments) / float(len(taxesupSentiments)-1)

	taxdown = 0 if (len(taxesdownSentiments) == 1) else reduce(lambda x, y: x + y, taxesdownSentiments) / float(len(taxesdownSentiments)-1)

	deathpenalty = 0 if (len(deathpenaltySentiments) == 1) else reduce(lambda x, y: x + y, deathpenaltySentiments) / float(len(deathpenaltySentiments)-1)

	healthcare = 0 if (len(healthcareSentiments) == 1) else reduce(lambda x, y: x + y, healthcareSentiments) / float(len(healthcareSentiments)-1)

	lgbt = 0 if (len(lgbtSentiments) == 1) else reduce(lambda x, y: x + y, gaySentiments) / float(len(gaySentiments)-1)

	environment = 0 if (len(environmentSentiments) == 1) else reduce(lambda x, y: x + y, environmentSentiments) / float(len(environmentSentiments)-1)

	welfare = 0 if (len(welfareSentiments) == 1) else reduce(lambda x, y: x + y, welfareSentiments) / float(len(welfareSentiments)-1)

	social = 0 if (len(socialSentiments) == 1) else reduce(lambda x, y: x + y, socialSentiments) / float(len(socialSentiments)-1)

	laborunion = 0 if (len(laborunionSentiments) == 1) else reduce(lambda x, y: x + y, laborunionSentiments) / float(len(laborunionSentiments)-1)

	drugs = 0 if (len(drugsSentiments) == 1) else reduce(lambda x, y: x + y, drugsSentiments) / float(len(drugsSentiments)-1)

	war = 0 if (len(warSentiments) == 1) else reduce(lambda x, y: x + y, warSentiments) / float(len(warSentiments)-1)
	

	print("Hillary sentiment: {}".format(hillary))
	print("Trump sentiment: {}".format(trump))
	print("Cruz sentiment: {}".format(cruz))
	print("Bernie sentiment: {}".format(bernie))
	print("Obama sentiment: {}".format(obama))
	print("Rpublican sentiment: {}".format(republican))
	print("Democrat sentiment: {}".format(democrat))
	print("Guns sentiment: {}".format(guns))
	print("Inmigration sentiment: {}".format(immigration))
	print("Employment sentiment: {}".format(employment))
	print("Inflation sentiment: {}".format(inflation))
	print("Minimum Wage Up sentiment: {}".format(wageup))	
	print("Abortion sentiment: {}".format(abortion))
	print("Govenment Spending sentiment: {}".format(govspend))
	print("Taxes Up sentiment: {}".format(taxup))
	print("Taxes Down sentiment: {}".format(taxdown))
	print('Death penalty sentiment: {}'.format(deathpenalty))
	print('Health care sentiment: {}'.format(healthcare))
	print('LGBT sentiment: {}'.format(lgbt))
	print('Environment sentiment: {}'.format(environment))
	print('Welfare sentiment: {}'.format(welfare))
	print('Social sentimnt: {}'.format(social))
	print('Labor Union sentiment: {}'.format(laborunion))
	print('Drugs sentiment: {}'.format(drugs))
	print('War sentiment: {}'.format(war))


	#COMPONENTS:

	#C1//FOR
	Xdeathpenalty = deathpenalty

	#C1//AGAINST
	Xhealthcare = -(healthcare)
	Xlgbt = -(lgbt)
	Xenvironment = -(environment)
	Xabortion = -(abortion)
	Xwelfare = -(welfare)
	Xsocial = -(social)
	Xlaborunion = -(laborunion)

	#CONTEMPLATED: deathpenalty, healthcare, lgbt, environment, abortion, welfare, social, laborunion.
	C1OUT = (Xdeathpenalty + Xhealthcare + Xlgbt + Xenvironment + Xabortion + Xwelfare + Xsocial + Xlaborunion) / 8

	#C2//FOR
	Ydrugs = drugs
	Yabortion = abortion
	Ylgbt = lgbt
	
	#C2//AGAINST
	Ywar = -(war)
	Ysocial = -(social)
	Yhealthcare = -(healthcare)
	Ywelfare = -(welfare)

	#CONTEMPLATED: drug, abortion, lgbt, war, social, healthcare, welfare
	C2OUT = (Ydrugs + Yabortion + Ylgbt + Ywar + Ysocial + Yhealthcare + Ywelfare) / 7
	#The conventional ideological buckets these positions fall into are social liberalism, fiscal conservatism, and opposition to foreign intervention- the primary features of libertarianism.


	img = plt.imread("Nolan_chart_normal.png")
	fig, ax = plt.subplots()
	ax.imshow(img, extent=[-1, 1, -1, 1])
	ax.plot(C1OUT, C2OUT, 'ro')
	plt.show()	
	
	if C1OUT > 0:
		bias = "right"
	elif C1OUT < 0:
		bias = "left"

	return bias



def send_pro_vote_propaganda(target): #TODO
	pass

def send_pro_protest_propaganda(target): #TODO
	pass



if __name__ == "__main__":

	banner()
	parsein()
	
#	users = pd.DataFrame()
#	users['user'] = 
#	users['id'] =
#	users['wing'] =
	
	benefit = BENEFIT
	num = NUM
	targets = TARGETS

	for target in targets:


		url = 'https://www.twitter.com/' + target	
		r = requests.get(url)
		soup = BeautifulSoup(r.content, "lxml")

		f = soup.find('li', class_="ProfileNav-item--followers").find('a')['title'] or ""
		num_followers = int(f.split(' ')[0].replace(',','').replace('.', ''))
		location = " ".join((soup.find('span', {'class': 'ProfileHeaderCard-locationText u-dir'}).text.replace('\n', '') or "").split())

		print("[*] Target: %s" % target)
		print("[*] Benefit %s wing" % benefit)
		print("[*] Collecting %s Tweets\n\n" % num)
		
		orientation = main(target, num);
		if orientation == bias:
			print("\nAction: Send Pro-Vote Propaganda.")
			send_pro_vote_propaganda(target)
		else:
			print("\nAction: Send Pro-Blank Propaganda.")
			send_pro_protest_propaganda(target)
			



"""
with open('train.json', 'r') as fp:
	cl = NaiveBayesClassifier(fp, format="json")

cl.classify("Very damm positive sentence")

prob_dist = cl.prob_classify("This one's a doozy.")
prob_dist.max()

round(prob_dist.prob("pos"), 2)

round(prob_dist.prob("neg"), 2)

from textblob import TextBlob
blob = TextBlob("The beer is good. But the hangover is horrible.", classifier=cl)
blob.classify()

for s in blob.sentences:
	print(s)
	print(s.classify())

cl.accuracy(test)
cl.show_informative_features(5)  

new_data = [('She is my best friend.', 'pos'),
                ("I'm happy to have a new friend.", 'pos'),
                ("Stay thirsty, my friend.", 'pos'),
                ("He ain't from around here.", 'neg')]
cl.update(new_data)

cl.accuracy(test)
"""


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

for tweet in tweets:
    detected = detect(tweet.text)
    cur.execute('''INSERT OR IGNORE INTO Tweets_GeoPLACE (
        user_id, user_name, user_timezone, user_language, detected_language, tweet_text, tweet_created
        ) 
    VALUES ( ?,?,?,?,?,?,? )''', (tweet.user.id,tweet.user.screen_name,tweet.user.time_zone,tweet.user.lang,detected,tweet.text,tweet.created_at))
    conn.commit()

from_sql = pd.read_sql_query("SELECT * FROM Tweets_GeoPLACE;", conn)
print(from_sql)
"""


"""
	mean = np.mean(data['len'])

	fav_max = np.max(data['Likes'])
	rt_max  = np.max(data['RTs'])

	fav = data[data.Likes == fav_max].index[0]
	rt  = data[data.RTs == rt_max].index[0]

	# Max FAVs:
	print("\nAverage tweet length: {}".format(mean))
	print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
	print("Number of likes: {}".format(fav_max))
	print("{} characters.\n".format(data['len'][fav]))

	# Max RTs:
	print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
	print("Number of retweets: {}".format(rt_max))
	print("{} characters.\n".format(data['len'][rt]))

	# create time series for data:

	tlen = pd.Series(data=data['len'].values, index=data['Date'])
	tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
	tret = pd.Series(data=data['RTs'].values, index=data['Date'])

	#tlen.plot(figsize=(16,4), color='r');
	#tlen.show()
	plt.plot(tfav, color='r')
	plt.plot(tfav, label="Likes", tret, label="Reweets", legend=True)
	plt.show()

"""


#NOLAN CHART DERIVATION

#Component 1 (X) Right hand edge of the plot.
""" 
for:
	death penalty

against: 
	National Health Care
	Gay Marriage
	Global Warming Exists
	Abortion
	Welfare
	Social Programs
	Medicaid & Medicare
	Environmental Protection
	Labor Union 
"""

#component 2 (Y) Top edge of the plot.
"""
for: 
	Drug Legalization
	Medical Marijuana
	Abortion
	Gay Marriage
against: 
	War on Terror
	Social Security
	War in Afghanistan
	Medicaid & Medicare
	Social Programs
	Welfare. 

The conventional ideological buckets these positions fall into are social liberalism, fiscal conservatism, and opposition to foreign intervention- the primary features of libertarianism.
"""





