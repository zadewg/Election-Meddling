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
from client import get_data
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

	print("\b                  Red Team Intelligence               ")

	print("\n     https://github.com/zadewg/Election-Meddling  \n\n")


def parsein():

	global TARGETS, NUM, BENEFIT

	parser = argparse.ArgumentParser(description='https://github.com/zadewg/Election-Meddling')
	parser.add_argument('-t','--target', help='Target sername(s)', required=True)
	parser.add_argument('-c','--count', help='Number of tweets to retrieve from user.', required=False)
	parser.add_argument('-b','--benefit', help='Political wing to benefit', required=True)
	args = vars(parser.parse_args())

	TARGETS = [args['target']]
	NUM = args['count'] or None
	BENEFIT = args['benefit']


class Splitter(object):

    def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self,text):

        sentences = self.splitter.tokenize(text)
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

	tweets = get_data(screenname, num)

	data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

	data['len']  = np.array([len(tweet.text) for tweet in tweets])
	data['ID']   = np.array([tweet.uid for tweet in tweets])
	data['Date'] = np.array([tweet.timestamp for tweet in tweets])
	data['Likes']  = np.array([tweet.likes for tweet in tweets])
	data['RTs']	= np.array([tweet.retweets for tweet in tweets])

	
	data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])

	display(data)

	pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
	neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
	neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]

	print("\n\nPercentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
	print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
	print("Percentage de negative tweets: {}%\n\n\n".format(len(neg_tweets)*100/len(data['Tweets'])))


#	for tweet in data['Tweets'].head(1):
#		print("\nORIGINAL SAMPLE: {}".format(tweet))
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
	republicanKeywords = ['republican', 'conservative', 'republican', 'republicans', 'republicanparty', 'republicandebate', 'republicansforberniesanders', 'republicanmemes', 'republicants', 'republicanlogic', 'republicanssuck', 'republicanhypocrisy', 'republicanprimary', 'republicansnowflakes', 'republicanos', 'republicanwomen', 'republicanbullshit', 'republicanvalues', 'republicanism', 'republicant', 'republicangirls', 'republicanpresidentialcandidate', 'republicanmountain', 'republicano', 'republicansaretheproblem', 'republicanera', 'republicanopub', 'republicanconvention', 'republicandebate', 'republicanforlife', 'republicansfortrump', 'republicansrule', 'republicanstrong', 'republicansunite', 'republicantaxplan', 'republicanthinking', 'republicantrump', 'republicanweed', 'republicanyonne', 'republicansforbernie', 'republicansarewhywecanthavenicethings', 'republicanarchitecture', 'republicanartist', 'republicanbabes', 'republicanclownshow', 'republicancorruption', 'republicandelegate', 'republicangirl', 'republicanidiot', 'republicaninstagram', 'republicanmedia', 'republicannominee', 'republicancandidate', 'republicanpride', 'republicanpropaganda', 'republicanrule', 'republicana']


	democratSentiments = [0]
	democratKeywords = ['democrat', 'dems', 'liberal', 'democracy', 'democrat', 'democrats', 'democratic', 'democraticparty', 'democraticsocialism', 'democraticsocialist', 'democraticdebate', 'democraticrepublicofcongo', 'democratssuck', 'democratie', 'democraticprimary', 'democratia', 'democraticconfederalism', 'democratsfortrump', 'democrata', 'democrate', 'democratgu', 'democrats', 'democratsabroad', 'democraticpresidentialdebate', 'democratization', 'democratizar', 'democraticnationalcommittee', 'democraticcamera', 'democratpoint', 'democratabarberclub', 'democraticrepublic', 'democratsneedtowin', 'democraticalliance', 'democratparty', 'democraticwhores', 'democratsareisis', 'democraticunionistparty', 'democraticsocislist', 'democratikdesign', 'democratsracist', 'democratsunite', 'democratsaredemons', 'democratizationofluxury', 'democratizeecommerce', 'democratsareracist', 'democratsinthehouse', 'democratizzare', 'democratizing', 'democraticnationalconvention', 'democraticcitizenship', 'democraticcaucus', 'democraticart', 'democraticaocialism', 'democratica', 'democratdonkey', 'democratconvention', 'democratbill', 'democratasontheradio', 'democratas', 'democratandchronicle', 'democraticdebate', 'democraticdecisionmaking', 'democraticschool', 'democratics', 'democraticrevolution', 'democraticrepublicanorwhatever', 'democraticreplublicofcongo', 'democraticreforms', 'democraticpeoplesrepublicofkorea', 'democraticpart', 'democraticownership', 'democraticnomination', 'democraticeducation', 'democratacalcados']

	
	gunsSentiments = [0]
	gunsKeywords = ['guns', 'gun', 'nra', 'pistol', 'firearm', 'shooting', 'gun', 'gunporn', 'guns', 'gunsdaily', 'gunsofinstagram', 'gunshow', 'gunstagram', 'gunslinger', 'gunsallowed', 'gunseason', 'gunsofig', 'gunshot', 'gunspictures', 'gunslifestyle', 'gunstore', 'gunsdailyusa', 'gunsafety', 'gunsout', 'gunsfanatics', 'gunsmith', 'gunsandammo', 'gunsandgirls', 'gunsbadassery', 'gunsforhands', 'gunsense', 'gunsup', 'gunscanada', 'gunsaz', 'gunsbitch', 'gunshop', 'gunsmithing', 'gunstonecreations', 'gunshots', 'gunstuff', 'gunsandcoffee', 'gunssavelives', 'gunsafe', 'gunsan', 'gunshotwound', 'gunsdaily1', 'gunsda', 'gunshinestate', 'gunsandbullets', 'gunsmithclothing', 'gunshooting', 'gunship', 'gunslingers', 'gunsdontkillpeople', 'gunspinner', 'gunsandgear', 'gunsandships', 'gunsofinsta', 'gunsofthepatriots', 'gunswag', 'gunsonpegs', 'gunslover', 'gunsmithsinhouston', 'gunslove', 'gunsfordays', 'gunsofboom', 'gunsails', 'gunshirt', 'gunsheesha', 'gunsales', 'gunsale', 'gunselfie', 'gunsallday', 'gunsuphoesdown', 'gunspredator', 'gunsrights', 'gunsmoke', 'gunsafemovers', 'gunsnothingelse', 'gunsinc', 'gunsplay', 'gunstock', 'gunsdrops', 'gunstoys', 'gunshirts', 'gunsyndicate', 'gunslife', 'gunsforsale', 'gunsofmayhem', 'gunsofsteel', 'gunsblazing', 'gunslingergirl', 'gunsandstuff', 'gunslings', 'gunsdrawn', 'gunsdownlifeup', 'gunstrokemc', 'gunstroke', 'gunsteel', 'gunskins', 'gunstrokesupplyco', 'gunsporn', 'gunsgunsguns', 'gunstyle', 'gunsnbutter', 'gunsonguns', 'gunsnbuns', 'gunsdaly', 'gunsmokebreakers', 'gunshu', 'gunshopping', 'gunsnrose', 'gunshoot', 'gunshotgoodbye', 'gunsshow', 'gunshootings', 'gunshy', 'gunsword', 'gunshows', 'gunsight', 'gunsjet', 'gunswithgirls', 'gunsofbrixton', 'gunsunderground', 'gunsuit', 'gunsofinstagrams', 'gunsontour', 'gunsorglitter', 'gunsquad', 'gunstagrammers', 'gunstocks', 'gunstonstreet', 'gunslikedave', 'gunsusa', 'gunsurvivor', 'gunsmithgirls', 'gunsnhoses', 'gunsmithstrength', 'gunsnammo', 'gunstarsuperheroes', 'gunstorage', 'gunsdonot', 'gunsandbutter', 'gunscarszombies', 'gunsandbuns', 'gunsandtactics', 'gunsandbeer', 'gunsammoamerica', 'gunsandwomans', 'gunsandwhiskey', 'gunsfishingandotherstuff', 'gunsandglory', 'gunscover', 'gunsandpearls', 'gunsandknives', 'gunsdownbikesup', 'gunsandhoses', 'gunseason', 'gunscreen', 'gunsandsavvy', 'gunsmylife', 'gunsgirls', 'gunsdontkillpeoplepeoplekillpeople', 'gunsaredrawn', 'gunsha', 'gunsgym', 'gunsgodgoverment', 'gunsafes', 'gunsgirlsandthegarage', 'gunsdays', 'gunsfordays']


	immigrationSentiments = [0]
	immigrationKeywords = ['immigration', 'immigrants', 'citizenship', 'naturalization', 'visas', 'immigrationclinic', 'immigrationconsulant', 'immigrationcontrol', 'immigrationdetention', 'immigrationfasttrack', 'immigrationhadley', 'immigrationhistory', 'immigrationtocanada', 'immigration', 'immigrationreform', 'immigrationlaw', 'immigrationlawyer', 'immigrationcanada', 'immigrationattorney', 'immigrationservices', 'immigrationterminal', 'immigrationprogram', 'immigrationmuseum', 'immigrationstories', 'immigrationrights', 'immigrationexpert', 'immigrationpanama', 'immigrationmedical', 'immigrationraids', 'immigrationteam', 'immigrationreformnow', 'immigrations', 'immigrationsolutionslawyers', 'immigrationspotlight', 'immigrationmarch', 'immigrationlaws', 'immigrationaction', 'immigrationadvisor', 'immigrationadvocate', 'immigrationanswers', 'immigrationappeals', 'immigrationattorneys', 'immigrationaustralia']


	employmentSentiments = [0]
	employmentKeywords = ['jobs', 'employment', 'unemployment', 'job']

	inflationSentiments = [0]
	inflationKeywords = ['inflate', 'inflation', 'price hike', 'price increase', 'prices rais']

	minimumwageupSentiments = [0]
	minimumwageupKeywords = ['raise minimum wage', 'wage increase', 'raise wage', 'wage hike', 'minimumwagedreamz', 'minimumwageincrease']

	abortionSentiments = [0]
	abortionKeywords = ['abortion', 'prochoice', 'plannedparenthood', 'abortion', 'abortionismurder', 'abortions', 'abortionrights', 'abortionaccess', 'abortionisnotmurder', 'abortioniswrong', 'abortionstigma', 'abortionkills', 'abortionists', 'abortionissin', 'abortionwithoutborders', 'abortionacesss', 'abortioncare', 'abortionstories', 'abortionhurtswomen', 'abortiondoula', 'abortionist', 'abortionenthusiast', 'abortionexploitswomen', 'abortionsurivor', 'abortionsupportnetwork', 'abortionsucks', 'abortionisacrime', 'abortionishealthcare', 'abortionalternatives', 'abortionrightsnow', 'abortionban', 'abortionpositive', 'abortionpill', 'abortionoptions', 'abortionislife', 'abortionmeme', 'abortionfunds', 'abortiondiscriminates', 'abortionisnotacrime', 'abortionondemand']


	governmentspendingSentiments = [0]
	governmentspendingKeywords = ['govspending', 'governmentspending', 'governmentspend', 'expenditure']

	taxesupSentiments = [0]
	taxesupKeywords = ['raisetax', 'taxhike', 'taxesup', 'taxup', 'increasetaxes', 'taxesincrease', 'taxincrease']

	taxesdownSentiments = [0]
	taxesdownKeywords = ['lowertax', 'taxcut', 'taxslash', 'taxesdown', 'taxdown', 'decreasetaxes', 'taxesdecrease', 'taxdecrease', 'taxesarescary', 'taxessuck', 'taxesarescary']
	
	deathpenaltySentiments = [0]
	deathpenaltyKeywords = ['deathrow', 'deathpenalty']

	healthcareSentiments = [0]
	healthcareKeywords = ['healthcare', 'healthcareforall', 'healthcaredesign', 'healthcaremarketing', 'healthcareprovider', 'healthcaremanagement', 'healthcarereform', 'healthcarecosts', 'healthcareassistant', 'healthcareadministration', 'healthcareprofessional', 'healthcarecostsareinsane', 'healthcareit', 'healthcarewithaheart', 'healthcareworkers', 'healthcareprofessionals', 'healthcarelife', 'healthcarepowerof', 'healthcarefinance', 'healthcareproduct', 'healthcareworker', 'healthcareequipment', 'healthcareexcellence', 'healthcareers', 'healthcaretech', 'healthcarejobs', 'healthcareservices', 'healthcarenotsickcare', 'healthcaremanager', 'healthcarecompliance', 'healthcareaide', 'healthcaresystem', 'healthcareinsurance', 'healthcareersdayonthehill', 'healthcareisahumanright', 'healthcarecost', 'healthcaregov', 'healthcareconnected', 'healthcarephoto', 'healthcareaustralia', 'healthcaresolutions', 'healthcarebranding', 'healthcaresocialmedia', 'healthcareproviders', 'healthcareisateamsport', 'healthcareseminar', 'healthcareproblems', 'healthcarepowerofattorney', 'healthcareapproved', 'healthcarematters', 'healthcaremarket', 'healthcareart', 'healthcareassistants', 'healthcareapp', 'healthcareforthehomeless', 'healthcareforeveryone', 'healthcareadmin', 'healthcarevoter', 'healthcarecrisis', 'healthcareconference', 'healthcare4all', 'healthcaretraining', 'healthcareevolution', 'healthcaretips', 'healthcarecentres', 'healthcarecenter', 'healthcarestaff', 'healthcaresucks', 'healthcareproxy', 'healthcarerebrand', 'healthcarequality', 'healthcareafrica', 'healthcareaintfree', 'healthcareshaklee', 'healthcareprograms', 'healthcareadvertising', 'healthcarewarriors', 'healthcaresuite', 'healthcaresales', 'healthcaretax', 'healthcareadvertisement', 'healthcaretime', 'healthcareselfcare', 'healthcarescience', 'healthcaresalesmentor', 'healthcareunit', 'healthcarereformsucks', 'healthcarevirtuallyanywhere', 'healthcarereformact', 'healthcarealternativesystems', 'healthcareproducttester', 'healthcareproducts', 'healthcareathletx', 'healthcareawards2017', 'healthcarecebu', 'healthcareaz', 'healthcarebpo', 'healthcaredubai', 'healthcaredotgovfailing', 'healthcaredollars', 'healthcaredesignation', 'healthcarecurrentevents', 'healthcarecoverage', 'healthcarebrooklyn', 'healthcarecareer', 'healthcareconsulting', 'healthcareconsultant', 'healthcarecompanion', 'healthcarefraud', 'healthcareindustry', 'healthcarejob', 'healthcareprobs', 'healthcarepoa', 'healthcareph', 'healthcarepenalty', 'healthcarepackaging', 'healthcareondemand', 'healthcareneeds', 'healthcareapps', 'healthcarearchitecture', 'healthcaremakeover', 'healthcareleadership', 'healthcareleader', 'healthcarejustice', 'healthcareassociatedinfections', 'healthcareisselfcare', 'healthcarecity', 'medicare', 'medicareforall', 'medicarepatients', 'medicare4all', 'medicaretreatment', 'medicareclinic', 'medicarechiropractor', 'medicarechiropractic', 'medicareaccepted', 'medicarevannuys', 'medicareaustralia', 'medicarehospital', 'medicareadvantage', 'medicarepharmacy', 'medicaresupplements', 'medicaid', 'medicaidcf', 'medicaidaccepted', 'medicaidplanning', 'medicaidreimbursement', 'medicaid4all', 'medicaidcancerfoundation', 'medicaidcancersummit', 'medicaidmess']

	
	lgbtSentiments = [0]
	lgbtKeywords = ['gay', 'lesbian', 'bisexual', 'transexual', 'lgbt', 'lgbtq', 'lgbtpride', 'lgbtcommunity', 'lgbtqia', 'lgbtqa', 'lgbtsupport', 'lgbtyouth', 'lgbtrights', 'lgbtpage', 'lgbtplus', 'lgbti', 'lgbtqpride', 'lgbttextposts', 'lgbtqiapd', 'lgbtlove', 'lgbtteens', 'lgbtfamily', 'lgbtequality', 'lgbtqi', 'lgbtaccount', 'lgbttravel', 'lgbtqcommunity', 'lgbta', 'lgbtposts', 'lgbtsupporter', 'lgbtqapride', 'lgbtqtravelers', 'lgbtqsupport', 'lgbtiq', 'lgbtqplus', 'lgbtqrights', 'lgbtteen', 'lgbtmemes', 'lgbtqiap', 'lgbtqap', 'lgbtcouple', 'lgbtsafezone', 'lgbtart', 'lgbtartist', 'lgbtwedding', 'lgbtqyouth', 'lgbthistory', 'lgbtsafe', 'lgbtproud', 'lgbthumor', 'lgbtsaga', 'lgbteens', 'lgbtmotivate', 'lgbthelp', 'lgbtlife', 'lgbtqiaplus', 'lgbtqlove', 'lgbtsafeplace', 'lgbttextpost', 'lgbtfilm', 'lgbtpride2017', 'lgbts', 'lgbtfiction', 'lgbtqai', 'lgbtatl', 'lgbtaq', 'lgbtqcute', 'lgbtqart', 'lgbtactivist', 'lgbtkids', 'lgbtfriendly', 'lgbtfamilies', 'lgbtqapd', 'lgbthistorymonth', 'lgbtitalia', 'lgbtqteens', 'lgbtuk', 'lgbtartists', 'lgbtqfilm', 'lgbtcouples', 'lgbtqtravel', 'lgbtally', 'lgbtparents', 'lgbtbooks', 'lgbtshoutout', 'lgbtunderground', 'lgbttravelers', 'lgbtweddings', 'lgbthiphop', 'lgbtquotes', 'lgbttumblr', 'lgbtacceptance', 'lgbtqpage', 'lgbtqfamily', 'lgbtt', 'lgbtnightlife', 'lgbtfobia', 'lgbtbusiness', 'lgbtgoals', 'lgbtworld', 'lgbtpoc', 'lgbtpoetry', 'lgbtqartist', 'lgbttravels', 'lgbtia', 'lgbtrightsarehumanrights', 'lgbtmusic', 'lgbtour', 'lgbtqaccount', 'lgbtsafespace', 'lgbtpost', 'lgbtturkey', 'lgbtpridemonth', 'lgbtqiapride', 'lgbtfitness', 'lgbtnews', 'lgbtnation', 'lgbtiqa', 'lgbtqpiad', 'lgbtadventure', 'lgbtflag', 'lgbtteenpage', 'lgbtbarber', 'lgbtbar', 'lgbtpower', 'lgbtyoutubers', 'lgbtsquad', 'lgbtqfashion', 'lgbtqwedding', 'lgbtqp', 'lgbttti', 'lgbtindia', 'lgbtqiacommunity', 'lgbtpodcast', 'lgbthomeless', 'lgbtselfie', 'lgbtforever', 'lgbtqatlanta', 'lgbtkickedout', 'lgbtqaplus', 'lgbtcameroon', 'lgbtcards', 'lgbtfit', 'lgbtqaip', 'lgbtgay', 'lgbtunited', 'lgbtcharacter', 'lgbtqsaga', 'lgbtgroup', 'lgbtpro', 'lgbtyoutuber', 'lgbtarg', 'lgbtpluspride', 'lgbtphotographer', 'lgbtqq', 'lgbtqfriendly', 'lgbtqhomeless', 'lgbtqhubcameroon', 'lgbtoc', 'lgbtcosplayer', 'lgbtrussia', 'lgbtpoet', 'lgbtpositivity', 'lgbtmodel', 'lgbtbrasil', 'lgbtqpia', 'lgbtbali', 'lgbtbabes', 'lgbtsociedade', 'lgbtitaly', 'lgbtqnightlife', 'lgbtspain', 'lgbtsouthafrica', 'lgbtspectrum', 'lgbtinstagram', 'lgbtqteen', 'lgbtqacommunity', 'lgbtqbosses', 'lgbtprom', 'lgbtqqiaa', 'lgbtp', 'lgbtnj', 'lgbtrp', 'lgbtmodels']

	
	environmentSentiments = [0]
	environmentKeywords = ['environment', 'environmental', 'environmentallyfriendly', 'environmentalist', 'environmentalism', 'environmentalscience', 'environments', 'environmentaleducation', 'environmentalart', 'environmentfriendly', 'environmentalportrait', 'environmentdesign', 'environmentart', 'environmentalfriendly', 'environmentalprotection', 'environmentalawareness', 'environmentaljustice', 'environmentalhealth', 'environmentally', 'environmentaldesign', 'environmentalengineering', 'environmentalsustainability', 'environmentalstudies', 'environmentallyconscious', 'environmentalpolicy', 'environmentalyfriendly', 'environmentartist', 'environmentallaw', 'environmentalists', 'environmentalissues', 'environmentalgraphics', 'environmentalmanagement', 'environmentaspolitics', 'environmentalactivism', 'environmentphotography', 'environmentalconservation', 'environmentallysafe', 'environmentprotection', 'environmentalprotectionagency', 'environmentalproblems', 'environmentalwellbeing', 'environmentalhumanities', 'environmentalillness', 'environmentalphotography', 'environmentalethics', 'environmentalchange', 'environmentaldestruction', 'environmentalclub', 'environmentaltoxins', 'environmentalimpact', 'environmentalterrorism', 'environmentday', 'environmentalplanning', 'environmentalportraiture', 'environmentalportraits', 'environmentalstewardship', 'environmentallyconcious', 'environmentconflictday', 'environmentalartist', 'environmentalartists', 'environmentminister', 'environmentlyfriendly', 'environmentalday', 'environmentalbranding', 'environmentalcare', 'environmentalcollective', 'environmentalfilm', 'environmentallyfriendlyprinting', 'environmentalgraphicdesign', 'environmentportrait', 'environmentalsociology', 'environmentalservices', 'environmentalsensitivity', 'environmentalradiation', 'environmentalremediation', 'environmentalrentalservices', 'environmentalyfriendlymemes', 'environmentalsensitivities', 'environmentalsciences', 'environmentalresponsibility', 'environmentstudy', 'environmentsarebeingdestroyed', 'environmentalsciencerules', 'environmentmeetshealth', 'environmentalstandards', 'environmentmatters', 'environmentdrawing', 'environmentaltransformation', 'environmentcontrolofwisconsin', 'environmentalstudy', 'environmentalurbanexercises', 'environmentalyconcious', 'environmentawareness', 'environmentandpublicworks', 'environmentaltoothbrush', 'environmentfriendlyproducts', 'environmentgirl', 'environmentalstudiesmajor', 'environmently', 'environmentissues', 'environmentialartificialgrass', 'environmentfurniture', 'environmentaltech', 'environmentfriendy', 'environmentalvirtueethics', 'environmentalaction', 'environmentaled', 'environmentaleducationiscool', 'environmentalenrichment', 'environmentalentrepreneurs', 'environmentalessences', 'environmentalexposure', 'environmentalfilmfestival', 'environmentalfolkart', 'environmentalfootprint', 'environmentalfriendlyart', 'environmentalfriendlyfashion', 'environmentalgood', 'environmentaldisaster', 'environmentaldegradation', 'environmentalallergies', 'environmentalarchitecture', 'environmentalartanddesign', 'environmentalartenvironmentalartanddesignenvironmentalartist', 'environmentalawarness', 'environmentalbiology', 'environmentalcertification', 'environmentalchemistforlife', 'environmentalconservationorganization', 'environmentalconservationstudent', 'environmentalconsulting', 'environmentaldata', 'environmentalgrowthmovement', 'environmentalhealthmatters', 'environmentallysustainable', 'environmentallytrendy', 'environmentalmetal', 'environmentalmonitoring', 'environmentalpark', 'environmentalpolitics', 'environmentalpollution', 'environmentalproducts', 'environmentalinjustice', 'environmentalpsychology', 'environmentalquality', 'environmentalquotes', 'environmentallysafecleaning', 'environmentallyresponsible', 'environmentalhealthstudent', 'environmentalheroes', 'environmentalimpactassessment', 'environmentalinitiatives', 'environmentalistdog', 'environmentalkids', 'environmentallearningcenter', 'environmentallifestyle', 'environmentalliteracy', 'environmentallycleangloves', 'environmentallyfriendlyjewelry', 'environmentallyfriendlyproducts', 'environmentalracism', 'globalwarming', 'globalwarmingisreal', 'globalwarmingawareness', 'globalwarmingfashion', 'globalwarmingsolving', 'globalwarmingriders', 'globalwarmingproblems', 'globalwarmingmum', 'globalwarmingishere', 'globalwarmingishappening', 'globalwarmingisahoax', 'globalwarminghoax', 'globalwarmingedition', 'globalwarmingdenial', 'globalwarmingsucks']



	welfareSentiments = [0]
	welfareKeywords = ['welfare', 'welfarerichent', 'welfarerich', 'welfarestate', 'welfaresystem', 'welfaretowork', 'welfareprojectsrl', 'welfareproject', 'welfareorganizations', 'welfareofchildren', 'welfarefirst', 'welfarecadillac', 'welfarebeerleague', 'welfarebear', 'welfarewagon']


	socialSentiments = [0]
	socialKeywords = ['socialsecurity', 'socialsecuritybenefits', 'socialsecuritynumber', 'socialsecurityadministration', 'socialsecuritycard', 'socialprogram']


	laborunionSentiments = [0]
	laborunionKeywords = ['laborunion', 'laborunions', 'laborrights', 'aborights', 'workerrights', 'workerights']


	drugsSentiments = [0]
	drugsKeywords = ['drugs', 'drugstore', 'drugstoremakeup', 'drugsarebad', 'drugstorebeauty', 'drugstorehaul', 'drugsforlove', 'drugstorecosmetics', 'drugsclothing', 'drugscity', 'drugsfree', 'drugsarebadmkay', 'drugstoreproducts', 'drugskill', 'drugsrp', 'drugsandalcohol', 'drugsdontwork', 'drugsjonas', 'drugstorenews', 'drugslutfollowtrain', 'drugsong', 'drugsarentcool', 'drugstoreskincare', 'drugsrehab', 'drugstoreglam', 'drugstorefoundation', 'drugsinmybody', 'drugsandway', 'drugsandsongs', 'drugstorefinds', 'drugstorewhore', 'drugsoflove', 'drugstorebrand', 'drugstoredupe', 'drugsrecovery', 'drugstoreshenanigans', 'drugscreening', 'drugsformugs', 'drugstores', 'drugsandmoney', 'drugsareoldnews', 'drugscenery', 'drugsarebadmmmkay', 'drugstoreessentials', 'drugstoremakeuplook', 'drugstorenude', 'drugstorebrands', 'drugstorebrushes', 'drugstorehk', 'drugstorebeat', 'drugsynthesis', 'drugsyoushouldtryit', 'drugstoo', 'drugstohell', 'drugstorecl', 'drugstorepallete', 'drugstorelook', 'drugstoreflatlay', 'drugstoremake', 'drugstorelipstick', 'drugstoreproduct', 'drugstorefind', 'drugswontbreakyourheart', 'drugstoredupes', 'drugswork', 'drugstoredeals', 'drugsxo', 'drugstest', 'drugstar', 'drugsabuse', 'drugsareforlosers', 'drugsaregood', 'drugsaretotallybadandall', 'drugsbandana', 'drugsbeforehoes', 'drugscam', 'drugscammer', 'drugscityfans', 'drugsarebetter', 'drugsarebadmmkay', 'drugsaesthetic', 'drugsafety', 'drugsandattics', 'drugsanddeli', 'drugsandhugs', 'drugsandwomen', 'drugsarebadforyoukids', 'drugsarebadkids', 'drugsforsale', 'drugsgta', 'drugshond', 'drugsonly', 'drugsontheway', 'drugsoutof10', 'drugspouch', 'drugsrockandroll', 'drugsruinfamilies', 'drugsruinlives', 'drugssexandhousemusic', 'drugsindenjeans', 'drugsof1ove', 'drugshots', 'drugsinc', 'drugslife', 'drugslove', 'drugslutgainplane', 'drugsmugglinsince09', 'drugsnhiphop', 'drugsniffingdog', 'drugssexandrocknroll']

	warSentiments = [0]
	warKeywords = ['war', 'iraqwar', 'afghanistanwar', 'afghanistan', 'iraq']



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
					('Employment', employmentSentiments, employmentKeywords), 
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
			stweet = tweet.replace(" ", "")
			for person in personSentimentList:
				for keyword in person[2]:
					if keyword in stweet:
						tb = textblob.TextBlob(data_preparation(tweet))
						person[1].append(tb.sentiment.polarity)
						MENTIONS += 1
						break

			for issue in issueSentimentList:
				for keyword in issue[2]:
					if keyword in stweet:
						tb = textblob.TextBlob(data_preparation(tweet))
						issue[1].append(tb.sentiment.polarity)
						MENTIONS += 1
						
					break

	except:
		print(sys.exc_info()[0])
		sys.stdout.flush()


	print("\nDetected {} mentions.\n".format(MENTIONS))


	
	hillary = 0 if (len(hillarySentiments) == 1) else reduce(lambda x, y: x + y, hillarySentiments) / float(len(hillarySentiments)-1)

	trump = 0 if (len(trumpSentiments) ==1) else reduce(lambda x, y: x + y, trumpSentiments) / float(len(trumpSentiments)-1)

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
	print("Immigration sentiment: {}".format(immigration))    
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



	C1ACC = 0
	#C1//FOR
	Xdeathpenalty = deathpenalty
	if str(immigration) != 0: 
		C1ACC += 1 

	Xguns = guns
	if str(guns) != "0":
		C1ACC += 1 

	Xtaxup = taxup
	if str(taxup) != "0":
		C1ACC += 1 

	Xtaxdown = taxdown
	if str(taxdown) != "0":
		C1ACC += 1 

	Xwar = war
	if str(war) != "0":
		C1ACC += 1 

	Xgovspend = govspend
	if str(govspend) != "0":
		C1ACC += 1 

	#C1//AGAINST
	Ximmigration = -(immigration)
	if str(immigration) != "0": 
		C1ACC += 1

	Xwageup = -(wageup)
	if str(wageup) != "0":
		C1ACC += 1 

	Xdrugs = -(drugs)
	if str(drugs) != "0":
		C1ACC += 1 

	Xhealthcare = -(healthcare)
	if str(healthcare) != "0":
		C1ACC += 1 

	Xlgbt = -(lgbt)
	if str(lgbt) != "0":
		C1ACC += 1 

	Xenvironment = -(environment)
	if str(environment) != "0":
		C1ACC += 1 

	Xabortion = -(abortion)
	if str(abortion) != "0":
		C1ACC += 1 

	Xwelfare = -(welfare)
	if str(welfare) != "0":
		C1ACC += 1 

	Xsocial = -(social)
	if str(social) != "0":
		C1ACC += 1 

	Xlaborunion = -(laborunion)
	if str(laborunion) != "0":
		C1ACC += 1 


	C1OUT = (Xdeathpenalty + Xguns + Xtaxup + Xtaxdown + Xwar + Xgovspend + Ximmigration + Xwageup + Xdrugs + Xhealthcare + Xlgbt + Xenvironment + Xabortion + Xwelfare + Xsocial + Xlaborunion) / C1ACC
	


	C2ACC = 0
	#C2//FOR
	Ydrugs = drugs
	if str(drugs) != "0":
		C2ACC += 1 

	Yabortion = abortion
	if str(abortion) != "0":
		C2ACC += 1 

	Ylgbt = lgbt
	if str(lgbt) != "0":
		C2ACC += 1 

	Yimmigration = immigration
	if str(immigration) != "0": 
		C2ACC += 1

	Ywageup = wageup
	if str(wageup) != "0":
		C2ACC += 1 

	Ytaxdown = taxdown
	if str(taxdown) != "0":
		C2ACC += 1 

	Yenvironment = environment
	if str(environment) != "0":
		C2ACC += 1 

	Ysocial = social
	if str(social) != "0":
		C2ACC += 1 

	Ylaborunion = laborunion
	if str(laborunion) != "0":
		C2ACC += 1 
	
	#C2//AGAINST
	Yguns = -(guns)
	if str(guns) != "0":
		C2ACC += 1 

	Ytaxup = -(taxup)
	if str(taxup) != "0":
		C2ACC += 1 

	Ydeathpenalty = -(deathpenalty)
	if str(deathpenalty) != "0":
		C2ACC += 1 

	Ywar = -(war)
	if str(war) != "0":
		C2ACC += 1 

	Yhealthcare = -(healthcare)
	if str(healthcare) != "0":
		C2ACC += 1 

	Ywelfare = -(welfare)
	if str(welfare) != "0":
		C2ACC += 1 

	Ygovspend = -(govspend)
	if str(govspend) != "0":
		C2ACC += 1 


	C2OUT = (Ydrugs + Yabortion + Ylgbt + Yimmigration + Ywageup + Ytaxdown + Yenvironment + Ysocial + Ylaborunion + Yguns + Ytaxup + Ydeathpenalty + Ywar + Yhealthcare + Ywelfare + Ygovspend) #C12ACC




	img = plt.imread("Nolan_chart_normal.png")
	fig, ax = plt.subplots()
	ax.imshow(img, extent=[-1, 1, -1, 1])
	ax.plot(C1OUT, C2OUT, 'ro')

	
	if C1OUT > 0:
		bias = "right"

	elif C1OUT < 0:
		bias = "left"


	return bias



def send_pro_vote_propaganda(target):
	pass

def send_pro_protest_propaganda(target):
	pass



if __name__ == "__main__":

	banner()
	parsein()
	
	benefit = BENEFIT
	
	num = int(NUM) if NUM != None else None
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
		print("[*] Collecting ~{} Tweets\n\n".format(num if num != None else 'all'))
			
		orientation = main(target, num);

		print("\n\n")

		if orientation == benefit:
			print("[*]Action: Send Pro-Vote Propaganda")
			send_pro_vote_propaganda(target)
		else:
			print("[*]Action: Send Pro-Blank Propaganda.")
			send_pro_protest_propaganda(target)

	plt.show()	#TODO: DYNAMIC PLOTTING
	



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



