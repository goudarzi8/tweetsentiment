import urbandictionary as ud
import numpy as np
import pandas as pn
import urlparse
import enchant
import nltk
import string
d = enchant.Dict("en_US")

def cleaner(tweet) :
    newTweet = ''
    try: 
        for i in tweet.split():
            s, n, p, pa, q, f = urlparse.urlparse(i)
            if s and n:
                pass
            elif i[:1] == '@':
                pass
            elif i[:1] == '#':
                newTweet = newTweet.strip() + ' ' + i[1:]
            else:
                newTweet = newTweet.strip() + ' ' + i
    except ValueError :
        newTweet = tweet
    return newTweet

def clear_punctuation(s):
    clear_string = ""
    for symbol in s:
        if symbol not in string.punctuation:
            clear_string += symbol
        else :
            clear_string += " "
    return clear_string

def dictionary(tweet) :
    tokens = nltk.word_tokenize(tweet)
    newTweet = ""
    for word in tokens:
        if(d.check(word)):
            newTweet +=  word
            newTweet +=  " "
        else :
            try :
                definition = str(ud.define(word)[0].definition[:40])
                if(definition == []) :
                    pass
                else :
                    newTweet += clear_punctuation(cleaner(definition))
                    newTweet +=  " "
            except Exception as e:
                print e
                newTweet +=  " "
    return newTweet

r_cols = ['sentiment','tweet']
data = pn.read_csv('golden-data.csv',sep=',',names=r_cols,usecols=range(2),na_filter=True,header=None)
data = data.dropna()

for i in range(len(data['tweet'].values)):
    print i
    try:
        data['tweet'].values[i] = dictionary(clear_punctuation(cleaner(data['tweet'].values[i]))) 
    except Exception as e:
        print e
        data['tweet'].values[i] = ""
data = data.dropna()
data.to_csv(path_or_buf = "fixed-golden-data.csv")
