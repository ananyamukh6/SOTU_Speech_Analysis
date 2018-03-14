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
import seaborn as sns; sns.set()
from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN

def kldiv(p,q):
    assert np.isclose(sum(p.values()),1) and np.isclose(sum(p.values()),1) and all([k>=0 for k in p.values()]) and all([k>=0 for k in q.values()]) #assert its a valid pdf
    fn = lambda x,y : -x*math.log(y/(x+0.0), 2)
    terms = {word:fn(p[word], q[word]) for word in p}
    #sort terms to find words that contribute most to the divergence
    words = [k for k in terms.keys()]
    sortidx = np.argsort([terms[w] for w in words])[-25:][::-1]
    return sum(terms.values()), [(words[idx], terms[words[idx]]) for idx in sortidx]
    
def get_pdf(corpus, vocab): #laplace smoothed. returns a dictionary of word->probability. the dict values sum to 1
    pdf = {w:1.0 for w in vocab}
    for w in corpus:
        pdf[w] += 1
    return {w:pdf[w]/sum(pdf.values()) for w in pdf}

def filter(words_in_speech):
    return [word for word in words_in_speech if word not in stopwords and word not in string.punctuation and '\'' not in word and len(word)>1]

def helper(presis, heatmapvals, presi_start_year):
    bigmap =  np.zeros([len(presis), len(presis)])
    for presiidx1, presi1 in enumerate(presis):
        print (presi1)
        for presiidx2, presi2 in enumerate(presis):
            try:
                #pdb.set_trace()
                nextpresi1 = presi_start_year[presis[presiidx1+1]]
            except:
                nextpresi1 = 2017
            try:
                nextpresi2 = presi_start_year[presis[presiidx2+1]]
            except:
                nextpresi2 = 2017
            #pdb.set_trace()
            bigmap[presiidx1][presiidx2] = np.mean(heatmapvals[presi_start_year[presi1]-1901:nextpresi1-1901, presi_start_year[presi2]-1901:nextpresi2-1901])
    return bigmap
    
def draw_heatmap(speechList):
    vocab = set([])
    for idx, speech in enumerate(speechList):
        words_in_speech = sum(speech['text_lem'], [])
        words_in_speech_filt = filter(words_in_speech)
        vocab = vocab.union(set(words_in_speech_filt))
    #vocab is a list of all unique words in all the speeches
    pdflist = {}
    try:
        pdflist = pkl.load(open('unigram_pdf.pkl', 'rb'))
    except:
        for idx, speech in enumerate(speechList):
            year = int(speech['date'].split(' ')[-1])
            print (year,'xx')
            pdflist[year] = [speech['speaker'], get_pdf(filter(sum(speech['text_lem'],[])), vocab)]
        pkl.dump(pdflist, open('unigram_pdf.pkl', 'wb'))
        
    try:
        heatmapvals = pkl.load(open('heatmap.pkl', 'rb'))
    except:
        heatmapvals = np.zeros([len(range(1901, 2017)), len(range(1901, 2017))])
        for year1 in range(1901, 2017):
            print(year1)
            for year2 in range(1901, 2017):
                #pdb.set_trace()
                try:
                    kl1, terms1 = kldiv(pdflist[year1][-1], pdflist[year2][-1])
                    kl2, terms2 = kldiv(pdflist[year2][-1], pdflist[year1][-1])
                    heatmapvals[year1-1901][year2-1901] = 0.5*(kl1+kl2)
                except:
                    continue
        pkl.dump(heatmapvals, open('heatmap.pkl', 'wb'))
        
    #create a (chronologically ordered) list of presidents, and their start years
    presis = []; presi_start_year = {}
    for year in range(1901, 2017):
        try:
            if pdflist[year][0] not in presis:
                presis += [pdflist[year][0]]
                presi_start_year[pdflist[year][0]] = year
        except:
            continue
    bigmap = helper(presis, heatmapvals, presi_start_year)  #of size num_presis x num_presis

    np.savetxt('heatmap.csv', heatmapvals, delimiter=',')       
    np.savetxt('heatmapbig.csv', bigmap, delimiter=',')  
    #pdb.set_trace()
    #sns.heatmap(heatmapvals)
    sns.heatmap(bigmap)
    plt.savefig('heatmapbig.png')
    plt.close()
    sns.heatmap(heatmapvals[-28:, -28:]) #bush senior to obama only
    plt.savefig('heatmapzoom.png')
    plt.close()
    #pdb.set_trace()
    
    
    
    
    
    
    from sklearn.decomposition import PCA
    
    keys = pdflist[2001][1].keys()
    unigramft = np.array([[pdflist[yr][1][k] for k in keys] for yr in range(1989, 2017)])
    #pdb.set_trace()
    pca = PCA(n_components=10)
    print ('start PCA')
    pca.fit(unigramft)
    print ('fitted PCA')
    newfts = pca.transform(unigramft)
    print ('transformed PCA')
    #pdb.set_trace()
    
    try:
        heatmapvals_zoom_pca = pkl.load(open('heatmap_zoom_pca.pkl', 'rb'))
    except:
        heatmapvals_zoom_pca = np.zeros([len(range(1989, 2017)), len(range(1989, 2017))])
        for year1 in range(1989, 2017):
            print(year1)
            for year2 in range(1989, 2017):
                #pdb.set_trace()
                try:
                    #pdb.set_trace()
                    f1 = newfts[year1-1989,:]
                    f2 = newfts[year2-1989,:]
                    heatmapvals_zoom_pca[year1-1989][year2-1989] = np.linalg.norm(f1-f2)
                except:
                    continue
        pkl.dump(heatmapvals_zoom_pca, open('heatmap_zoom_pca.pkl', 'wb'))
        
    sns.heatmap(heatmapvals_zoom_pca) #bush senior to obama only
    plt.savefig('heatmap_zoom_pca.png')
    plt.close()
    
    k_tr, dim_out, max_iter = 3, newfts.shape[1], 180
    clf = LMNN(n_neighbors=k_tr, max_iter=max_iter, n_features_out=dim_out, verbose=False)
    class_labels = [0]*3 + [1]*8 + [2]*9 + [3]*8
    clf = clf.fit(newfts, class_labels)
    
    ftmap = 0.1*np.ones([newfts.shape[0],newfts.shape[0]]) #28 x 28
    for idx1 in range(newfts.shape[0]):
        for idx2 in range(newfts.shape[0]):
            if idx1!=idx2:
                ftmap[idx1,idx2] = np.linalg.norm(clf.transform([newfts[idx1,:]]) - clf.transform([newfts[idx2,:]]))
    sns.heatmap(np.log(ftmap))
    plt.savefig('heatmap_pca_metric.png')
    plt.close()


listOfLines = picklines(open('stateoftheunion1790-2016.txt'), (range(90737, 180008+1))) #listOfLines is a list of strings. (separated by newline)
try:
    speechList = pkl.load(open('senti.pkl', 'rb'))
except:
    speechList = getSpeechList(listOfLines)
    pkl.dump(speechList, open('senti.pkl', 'wb'))
draw_heatmap(speechList)
