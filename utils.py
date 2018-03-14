import sys, re, string, nltk, numpy as np
from nltk.collocations import *
#from sets import Set
from scipy import stats
import math, pdb
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer


stopwords = ['must','we','is','are','t','ll','shall','whose','will','a','about','above','after','again','against','all','also','am','an','and','any','are','aren\'t','as','at','be','because','been','before','being','below','between','both','but','by','can\'t','cannot','could','couldn\'t','did','didn\'t','do','does','doesn\'t','doing','don\'t','down','during','each','few','for','from','further','had','hadn\'t','has','hasn\'t','have','haven\'t','having','he','he\'d','he\'ll','he\'s','her','here','here\'s','hers','herself','him','himself','his','how','how\'s','i','i\'d','i\'ll','i\'m','i\'ve','if','in','into','is','isn\'t','it','it\'s','its','itself','let','let\'s','me','more','most','mustn\'t','my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours','ourselves','out','over','own','same','shan\'t','she','she\'d','she\'ll','she\'s','should','shouldn\'t','so','some','such','than','that','that\'s','the','their','theirs','them','themselves','then','there','there\'s','these','they','they\'d','they\'ll','they\'re','they\'ve','this','those','through','to','too','under','until','up','very','was','wasn\'t','we','we\'d','we\'ll','we\'re','we\'ve','were','weren\'t','what','what\'s','when','when\'s','where','where\'s','which','while','who','who\'s','whom','why','why\'s','with','won\'t','would','wouldn\'t','you','you\'d','you\'ll','you\'re','you\'ve','your','yours','yourself','yourselves']

def picklines(file, lines):
    return [x for i, x in enumerate(file) if i in lines]
    
def processText1(text): #strip the lines of \n, join them, split them on line enders, also converts to lowercase. Note this function returns strings that contain punctuations    
    alltext = ' '.join([itr.strip().lower() for itr in text])
    sent_tokenize_list = sent_tokenize(alltext) # doc is a list of sentences
    word_tokenized = [nltk.word_tokenize(sent) for sent in sent_tokenize_list]  #list of sentences -> list of words
    
    #stem = PorterStemmer().stem
    lem = WordNetLemmatizer().lemmatize
    #stemmed = [[stem(word) for word in sent] for sent in word_tokenized] 
    lemmed = [[lem(word) for word in sent] for sent in word_tokenized]
    #return sent_tokenize_list, word_tokenized, stemmed, lemmed
    return word_tokenized, lemmed
    
def getSpeechList(listOfLines): #Divide the speeches by speaker
    speakerList = []; dateList = []; startlineList = []
    for linecount in range(0, len(listOfLines)):
        #pdb.set_trace()
        if '***\n' == listOfLines[linecount]:  #using *** to split speeches
            speakerList.append(listOfLines[linecount + 3].strip())
            dateList.append(listOfLines[linecount + 4].strip())
            startlineList.append(linecount + 6)
    startlineList.append(int(180008))
    


    #print (speakerList)
    #print (dateList)
    #print (startlineList)
    partyDict = {'George H.W. Bush' : 'R', 'William J. Clinton' : 'D', 'George W. Bush' : 'R', 'Barack Obama' : 'D', 'Ronald Reagan': 'R', 'Jimmy Carter':'D', 'Gerald R. Ford':'R', 'Richard Nixon':'R', 'Lyndon B. Johnson':'D', 'John F. Kennedy':'D', 'Dwight D. Eisenhower':'R', 'Harry S. Truman':'D', 'Franklin D. Roosevelt':'D', 'Herbert Hoover':'R', 'Calvin Coolidge':'R', 'Warren Harding':'R', 'Woodrow Wilson':'D', 'William H. Taft':'R', 'Theodore Roosevelt':'R'}
    speechList = []  #a list of dictionaries, each dictionary is like a structure representing a speech
    for speechCount in range(0, len(dateList)):
        speech = {}
        speech['speaker'] = speakerList[speechCount]; speech['date'] = dateList[speechCount]; speech['party'] = partyDict[speakerList[speechCount]]
        speech['text'], speech['text_lem'] = processText1(listOfLines[startlineList[speechCount] : startlineList[speechCount+1] - 6])  #the -6 skips the header of the next speech
        speechList.append(speech)

    return speechList
    
def getCorpus(speechList, party ):  #party can be 'R', 'D' or 'A' (A for all).
    corpus = []
    for speech in speechList:
        if (party == 'R' and speech['party'] == 'R') or (party == 'D' and speech['party'] == 'D') or (party == 'A'):
            corpus.extend(speech['text'])
    return ('. '.join(corpus))

    

def findBigrams(text, measure):
    tokens = nltk.wordpunct_tokenize(text)
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(3)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    measureUsed = {'raw_freq' : bigram_measures.raw_freq, 'pmi' : bigram_measures.pmi, 'chi_sq' : bigram_measures.chi_sq, 'likelihood_ratio' : bigram_measures.likelihood_ratio}.get(measure, None)
    scored = finder.score_ngrams(measureUsed)
    scoredFiltered = []
    for itr in scored:
        pair = itr[0]
        if pair[0] == '.' or pair[1] == '.':
            continue
        if pair[0] in stopwords or pair[1] in stopwords:
            continue
        if pair[0].isdigit() or pair[1].isdigit():
            continue
        scoredFiltered.append(itr)
        print (scoredFiltered)
        a = [itr[1] for itr in sorted((e[1],i) for i,e in enumerate(scoredFiltered))]
        return print ([' '.join(scoredFiltered[i][0]) for i in a][-25:]) #sort by score (e[1]), then pick out the indices (of the sorted list).  sorting occurs in ascending order, so [-25:] selects the last 25 elements
    
def intersectSets(setlist):  #print intersections of all tuples of sets in the list
    for i in range(0, len(setlist)):
        for j in range(i+1, len(setlist)):
            print (setlist[i].intersection(setlist[j]))
            