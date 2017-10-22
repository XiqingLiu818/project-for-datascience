#1a,
import pandas as pd
import numpy as np
from sklearn import feature_extraction
from scipy import stats
from sklearn import decomposition,linear_model
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier,Lasso,SGDClassifier,LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score,hamming_loss
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import pandas as pd
import numpy
import nltk
import os
import re
from nltk.corpus import stopwords,wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn import decomposition
import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

thePath = '/Users/apple/Desktop/project/classify/'
thePathLut = '/Users/apple/Desktop/project/classify/'
theCols = os.walk(thePath).next()[1]  

theLabels = theCols 

finalWords = list()
theDocs = list()

def genCorpus(theText):
    #set dictionaries
    stopWords = set(stopwords.words('english'))
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

def textToNum(theLabels,thePredLabel):
    theOutLabel = dict()
    cnt = 0
    for word in theLabels:
        theOutLabel[word] = cnt
        cnt = cnt + 1
    return str(theOutLabel[thePredLabel])

theLUT = pd.read_csv(thePathLut + 'classifierLUT.csv',index_col=0) #ALGO LUT
def optFunc(theAlgo,theParams):
    theModel = theLUT.loc[theAlgo,'optimizedCall']
    tempParam = list()
    for key, value in theParams.iteritems():
        tempParam.append(str(key) + "=" + str(value)) 
    theParams = ",".join(tempParam)
    theModel = theModel + theParams + ")"
    return theModel 

def algoArray(theAlgo):
    theAlgoOut = theLUT.loc[theAlgo,'functionCall']
    return theAlgoOut

for word in theCols:
    cnt = 0
    for file in os.listdir(thePath+word):
        if file.endswith('.txt'):
            try:
                f = open(thePath + word + "/" + file, "r")
                lines = f.readlines()
                lines = [text.strip() for text in lines]
                lines = " ".join(lines)
                finalWords.append(genCorpus(lines))
                theDocs.append(textToNum(theLabels,word) +"_" + str(cnt))
                cnt = cnt +  1
            except:
                pass
            
vectorizer = TfidfVectorizer(max_features=1000,ngram_range=(1,1))
tdm = pd.DataFrame(vectorizer.fit_transform(finalWords).toarray())

with open('vectorizer.pk', 'wb') as fin:
    pickle.dump(vectorizer, fin)

tdm.columns=vectorizer.get_feature_names()
tdm.index=theDocs

pca = decomposition.PCA(n_components=.95)
pca.fit(tdm)
reducedTDM = pd.DataFrame(pca.transform(tdm)) #reduced tdm distance matrix

with open('pca.pk', 'wb') as fin:
    pickle.dump(pca, fin)

reducedTDM.index=theDocs

pcaVar = round(sum(pca.explained_variance_ratio_),2)

fullIndex = reducedTDM.index.values
fullIndex = [int(word.split("_")[0]) for word in fullIndex]

theModels = ['RF','LDA','NN','DT','ABDT']
theResults = pd.DataFrame(0,index=theModels,columns=['accuracy','confidence','runtime'])
for theModel in theModels:
    startTime = time.time()
    model = eval(algoArray(theModel))
    #model = RandomForestClassifier(random_state=50)
    print(theModel)

    #cross validation    
    cvPerf = cross_val_score(model,reducedTDM,fullIndex,cv=10)
    theResults.ix[theModel,'accuracy'] = round(cvPerf.mean(),2)
    theResults.ix[theModel,'confidence'] = round(cvPerf.std() * 2,2)
    endTime = time.time()
    theResults.ix[theModel,'runtime'] = round(endTime - startTime,0)
    
print(theResults)
#NN with accuracy 0.97

#############################################
#######Run with best performing model########
#####Fine Tune Algorithm Grid Search CV######
#############################################
bestPerfStats = theResults.loc[theResults['accuracy'].idxmax()]
modelChoice = theResults['accuracy'].idxmax()
              
startTime = time.time()
model = eval(algoArray(modelChoice))
grid = GridSearchCV(estimator=model, param_grid={"alpha": [1,0.1,0.01,0.001,0.0001,0]})#eval(gridSearch(modelChoice))
grid.fit(reducedTDM,fullIndex)
#grid.fit(train,trainIndex)
bestScore = round(grid.best_score_,4)
parameters = grid.best_params_
endTime = time.time()
print("Best Score: " + str(bestScore) + " and Grid Search Time: " + str(round(endTime - startTime,0)))

############################################
######Train Best Model on Full Data Set#####
########Save Model for future use###########
############################################
startTime = time.time()
model = eval(optFunc(modelChoice,parameters)) #train fully validated and optimized model
model.fit(reducedTDM,fullIndex)
#model.fit(train,trainIndex)
joblib.dump(model, modelChoice + '.pkl') #save model
endTime = time.time()
print("Model Save Time: " + str(round(endTime - startTime,0)))






#1b,
import tweepy  
from pymongo import MongoClient
from textwrap import TextWrapper
from tweepy.utils import import_simplejson
json = import_simplejson()

auth1 = tweepy.auth.OAuthHandler('3PnOT7I8YWhhWAUcrTdzYOSh4','	Irg9syhSXXyhtX17pt9w7lReYNzTgYDhwnEGig0PWT2gF76Gpb')  
auth1.set_access_token('913440309241171970-YIq2R7FmRGRuB7MlW5lXFcDRvX9GKlA','oVK28kQjXUBCFqX51ci1hOtRykfYuSfZdlBebZtWGvIxY')  
api = tweepy.API(auth1)

mongo = MongoClient('localhost', 27017)
mongo_db = mongo['twitterDBs']
mongo_collection = mongo_db['theData']

class StreamListener(tweepy.StreamListener):  
    status_wrapper = TextWrapper(width=140, initial_indent='', subsequent_indent='')    
    def on_status(self, status): 
        tempA = self.status_wrapper.fill(status.text)
        tempB = status.retweeted 
        tempC = status.user.lang 
        tempD = status.geo
        print tempD
        if ((("en" in tempC) and (tempB is False)) and (not("RT") in tempA[:2]) and (((("http" or "www") in tempA) and ((' ') in tempA)) or (not("http" or "www") in tempA))):
            try:     
                print(self.status_wrapper.fill(status.text))
                mongo_collection.insert({
                'body': self.status_wrapper.fill(status.text),
                'topic': self.bestmodel(status.text),
                'followers': status.user.followers_count,
                'screen_name': status.author.screen_name,
                'friends_count': status.user.friends_count,
                'created_at': status.created_at,
                'message_id': status.id,
                'location': status.user.location
                })
            except Exception, (e):  
                print("HERE")          
                pass 

    def genCorpus(self, theText):
        #set dictionaries
        stopWords = set(stopwords.words('english'))
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
    
    def BestModel(self, tempText):
        
        path_topic = '/Users/apple/Desktop/project/classify/'
        theCols = os.walk(path_topic).next()[1] 
        path_classifier = '/Users/apple/'
        vectorizer = joblib.load(path_classifier + 'vectorizer.pk') 
        pca = joblib.load(path_classifier + 'pca.pk') 
        model = joblib.load(path_classifier + 'NN.pkl')
        testText = list()
        testText.append(self.genCorpus(tempText))
        test = vectorizer.transform(testText)
        X2_new = pca.transform(test.toarray())
        x = model.predict(X2_new)[0]
        Cols = ["fishing","hiking","machine learning","mathematics"]
        
        #return the topic of tweets
        return Cols[x]



#1c,    
l = StreamListener()  
streamer = tweepy.Stream(auth=auth1, listener=l, timeout=3000)   
setTerms = ["fishing","hiking","machine learning","mathematics"]
streamer.filter(None,setTerms) 
    
    
        
    
