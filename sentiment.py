import sys, re, string, nltk, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.collocations import *
#from sets import Set
from scipy import stats
import math, pdb, pickle as pkl
from nltk.stem import WordNetLemmatizer
from utils import *

def plot_helper(sentiScore, party, nn, term):
    fig = plt.figure()
    ax = fig.gca()
    vals = [sentiScore[i][nn] for i in range(1901, 2017)]
    p10 = np.percentile(vals, 10)
    p90 = np.percentile(vals, 90)
    
    idx = np.argsort(vals)
    negyears = ([i+1901 for i in idx[:10]])
    posyears = ([i+1901 for i in idx[::-1][:10]])
    
    barlist = plt.bar(range(1901, 2017), vals, edgecolor='k')
    [barlist[i-1901].set_color('r') for i in party if party[i]=='R']
    [barlist[i-1901].set_color('b') for i in party if party[i]=='D']
    lastparty = party[1901]; presinum = 0; lastpresi = term[1901]
    coldictR = {0:'r', 1:'tomato', 2:'lightsalmon'}
    coldictD = {0:'darkblue', 1:'mediumblue'}
    for i in party:
        if party[i]=='R':
            if lastparty == 'D': #a change from 'D' to 'R':
                presinum = 0
                lastpresi = term[i]
            #barlist[i-1901].set_color(coldictR[presinum])
        else:
            if lastparty == 'R': #a change from 'R' to 'D':
                presinum = 0
                lastpresi = term[i]
            #barlist[i-1901].set_color(coldictD[presinum])
        if lastpresi != term[i]:
            presinum += 1
        if party[i]=='R':
            barlist[i-1901].set_color(coldictR[presinum])
        else:
            barlist[i-1901].set_color(coldictD[presinum])
        lastpresi = term[i]
        lastparty = party[i]
    pdb.set_trace()
    cnst = {0:0.002, 5:0.02}[nn]
    for i, item in enumerate(vals):
        if i+1901 in posyears:
            ax.plot([i+1901, i+1901], [-cnst, cnst], 'g')#, 'go', markersize=3)
        if i+1901 in negyears:
            ax.plot([i+1901, i+1901], [-cnst, cnst], 'k')#, 'ko', markersize=3)
    pdb.set_trace()
    plt.plot()
    #plt.grid()
    plt.savefig('sentiment'+str(nn)+'.png')
    print ('posyears', posyears)
    print ('negyears', negyears)
    
def createSentiDict():
    lem = WordNetLemmatizer().lemmatize
    lines = [line.strip().split('\t') for line in open('AFINN-111.txt').readlines()]
    return {lem(i[0]):int(i[1]) for i in lines}
    
def analyseSenti(speechList, sentiDict):
    sentiScore = {}
    repList = []; demList = []
    party = {}
    term = {}
    for speechID in range(len(speechList)):
        speech = speechList[speechID]
        #print (speech['date'])
        #pdb.set_trace()
        allWords = sum(speech['text_lem'],[])
        scoreList = [sentiDict.get(word,0) for word in allWords]
        year = int(speech['date'].split(' ')[-1])
        party[year] = speech['party']
        posScores = [i for i in scoreList if i>0]; negScores = [i for i in scoreList if i<0]
        sentiScore[year] = [np.mean(scoreList), len(posScores), len(negScores), np.mean(posScores), np.mean(negScores), np.mean([i for i in scoreList if i!=0 ])]
        term[year] = speech['speaker']
    
    for yr in range(1901,2017): #fill missing entries
        if yr not in sentiScore:
            sentiScore[yr] = [0]*6
    
    plot_helper(sentiScore, party, 0, term)
    plot_helper(sentiScore, party, 5, term)
    



    
#sentiment:
sentiDict = createSentiDict()   #163011
listOfLines = picklines(open('stateoftheunion1790-2016.txt'), (range(90737, 180008+1))) #listOfLines is a list of strings. (separated by newline)
#speechList = getSpeechList(listOfLines)
#pdb.set_trace()
try:
    speechList = pkl.load(open('senti.pkl', 'rb'))
except:
    speechList = getSpeechList(listOfLines)
    pkl.dump(speechList, open('senti.pkl', 'wb'))
analyseSenti(speechList, sentiDict)

