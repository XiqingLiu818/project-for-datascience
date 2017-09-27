#1a. Print all words beginning with sh.
sent = ['she', 'sells', 'see', 'shells', 'by', 'the', 'sea', 'shore']
for word in sent:
    if word[0] == 's' and word[1] == 'h':
        print word

#1b.Print all words longer than four characters. 
for word in sent:
    if len(word) > 4:
        print word
        
#2.The text “Wild brown trout are elusive“? Provide code that will append each word to a list
#Method 1:
text = 'Wild brown trout are elusive'
list1 = text.split()
print list1
#Method 2:
list2 = []
list2.append('Wild')
list2.append('brown')
list2.append('trout')
list2.append('are')
list2.append('elusive')
print list2

#3.
#First import the packages
import sklearn
import pandas as pd
import numpy
import nltk
import os
import re
from nltk.corpus import stopwords,wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

#to read the four topics
thePath = '/Users/apple/Desktop/projectstack/classify/'
topics = os.walk(thePath).next()[1]

def genCorpus(theText):
#set dictionaries
    stopWords = set(stopwords.words('english'))#words dont have meaning
    theStemmer = nltk.stem.porter.PorterStemmer() #Martin Porters celebrated stemming algorithm
    
    #pre-processing
    theText = theText.split()
    tokens = [token.lower() for token in theText] #ensure everything is lower case
    tokens = [re.sub(r'[^a-zA-Z0-9]+', ' ',token) for token in tokens] #remove special characters but leave word in tact
    tokens = [token for token in tokens if token.lower().isalpha()] #ensure everything is a letter
    tokens = [word for word in tokens if word not in stopWords] #rid of stop words
    tokens = [theStemmer.stem(word) for word in tokens] #stem words uing porter stemming algorithm
    tokens = " ".join(tokens) #need to pass string seperated by spaces
    
    return tokens

for word in topics:
    Final = [] 
    for file in os.listdir(thePath+word):
        if file.endswith('.txt'):
            try:
                f = open(thePath + word + "/" + file, "r") #open the txt
                lines = f.readlines()
                lines = [text.strip() for text in lines]
                lines = " ".join(lines)
                Final.append(genCorpus(lines))
            except:
                pass
    WordFre = {}
    for a in Final:
        wordinlist = str(a).split(" ") #change the data into str and split
        for b in wordinlist:
            if b in WordFre:
                WordFre[b] = WordFre[b] + 1 #add one frequency
            else: WordFre[b] = 1 #new word

    dataframe = pd.DataFrame(WordFre.items(), columns=['word','freq']) #dataframe
    dataframe.set_index('word', inplace = True) #set index as word
    dataframe.sort_values('freq', ascending = 0, inplace = True) #order the values
    dataframe.to_csv('/Users/apple/Desktop/'+ word+'.csv') #get csv
