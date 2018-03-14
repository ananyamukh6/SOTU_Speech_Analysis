import sys, re, string, nltk, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.collocations import *
#from sets import Set
from scipy import stats
import math, pdb, pickle as pkl, re
from nltk.stem import WordNetLemmatizer
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns; sns.set()
from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN
from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN

def display_topics(model, feature_names, no_top_words, possible_topics):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d: %s" % (topic_idx, possible_topics[topic_idx]))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def find_topic(speechList):
    speechList = speechList[-28:]  #from bush senior

    documents = []
    for speech in speechList:
        tmp = []
        for sent in speech['text_lem']:  #sent is a single sentence which is a list of words
            ss = []
            for w in sent: #w is a single word
                w = re.compile('[%s]' % re.escape(string.punctuation)).sub('', w) #replace punctuations in a word by ''
                if len(w)>2:
                    try:
                        float(w) #do not consider numbers
                    except:
                        ss += [w]
            tmp += [' '.join(ss)] #tmp is a list of sentences. sentence is a string (space separated words). tmp is a collection of sentences from a speech
        documents += [tmp]
    documents_sents = sum(documents,[]) #list of sentences of all speeches

    
    print(len(documents_sents))
    
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=2000, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents_sents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    num_topics = 10
    lda = LatentDirichletAllocation(n_topics=num_topics, max_iter=10, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    
    possible_topics = ['education', 'jobs', 'world affairs', 'health care', 'middle east', 'terrorism', 'taxation', 'social programs', 'law and order', 'iraq war'] #from later analysis
    
    display_topics(lda, tf_feature_names, 15, possible_topics)
    
    #now that we have the topic model, let us see, how each speech fares:
    topwords = []
    for topic_idx, topic in enumerate(lda.components_):
        #topic is a np array of 2000 (num features) numbers
        tmp = topic.argsort()[:-15 - 1:-1]
        topwords.append({tf_feature_names[i]:topic[i] for i in tmp})
    #topwords is of length #topics. topwords[i] is the top words (and scores) for topic i
    
    all_doc_fts = []
    for docidx, doc in enumerate(documents):
        doc_ft = []
        for sentidx, sent in enumerate(doc):
            words = sent.split(' ')
            topic_ft = [0]*len(topwords)  #length 10
            for word in words:
                for tp_idx, tp in enumerate(topwords):
                    if word in tp:
                        topic_ft[tp_idx] += 1
            doc_ft += [topic_ft] #doc_ft is num_of_sentences_in_doc x num_topics
        all_doc_fts += [doc_ft]
        
    finaldocft = np.array([np.mean(np.array(all_doc_ft), 0) for all_doc_ft in all_doc_fts])  #shape: 28 x 10 (num docs x num topics)
    
    speechinfo = [(speech['speaker'], speech['party']) for speech in speechList]
    top_speeches_using_topic = []
    #who used topic i the most?
    for topicidx in range(len(topwords)):
        tmp = np.argsort(finaldocft[:,topicidx])[::-1][:5]
        top_speeches_using_topic.append(tmp) #top 3 speeches that use this topic
        tmp1 = []
        for t in tmp:
            if speechinfo[t] not in tmp1:
                tmp1 += [speechinfo[t]]
        print ('Topic '+str(topicidx) + ': ' + possible_topics[topicidx])
        #tmp1 = (set([speechinfo[t] for t in tmp]))
        print([i[0] + ' (' + i[1] + ')' for i in tmp1])
        
    #each speech used which topics (top 3 topics per speech)?
    for idx, (speaker, party) in enumerate(speechinfo):
        tmp = np.argsort(finaldocft[idx,:])[::-1][:3]
        tp = ' '.join(['Topic ' + str(i)+ ' (' + possible_topics[i] + ')' for i in tmp])
        print (speaker + ' used the following topics: ' + tp)

    #pdb.set_trace()
    ftmap = 0.1*np.ones([finaldocft.shape[0],finaldocft.shape[0]]) #28 x 28
    for idx1 in range(finaldocft.shape[0]):
        for idx2 in range(finaldocft.shape[0]):
            if idx1!=idx2:
                ftmap[idx1,idx2] = np.linalg.norm(finaldocft[idx1,:]-finaldocft[idx2,:])
    sns.heatmap(np.log(ftmap))
    plt.savefig('heatmap_topic.png')
    plt.close()
            
    
    #Now do metric-learning
    k_tr, dim_out, max_iter = 3, finaldocft.shape[1], 180
    clf = LMNN(n_neighbors=k_tr, max_iter=max_iter, n_features_out=dim_out, verbose=False)
    class_labels = [0]*3 + [1]*8 + [2]*9 + [3]*8
    clf = clf.fit(finaldocft, class_labels)
    #accuracy_lmnn = clf.score(finaldocft, class_labels)
    #print ('Metric learn accuracy: ', accuracy_lmnn)
    
    ftmap = 0.1*np.ones([finaldocft.shape[0],finaldocft.shape[0]]) #28 x 28
    for idx1 in range(finaldocft.shape[0]):
        for idx2 in range(finaldocft.shape[0]):
            if idx1!=idx2:
                ftmap[idx1,idx2] = np.linalg.norm(clf.transform([finaldocft[idx1,:]]) - clf.transform([finaldocft[idx2,:]]))
    sns.heatmap(np.log(ftmap))
    plt.savefig('heatmap_topic_metric.png')
    plt.close()
    
    
    
    
    
    

listOfLines = picklines(open('stateoftheunion1790-2016.txt'), (range(90737, 180008+1))) #listOfLines is a list of strings. (separated by newline)
try:
    speechList = pkl.load(open('senti.pkl', 'rb'))
except:
    speechList = getSpeechList(listOfLines)
    pkl.dump(speechList, open('senti.pkl', 'wb'))
find_topic(speechList)
