
from gensim import corpora, models, similarities
import csv
import numpy as np

#Load docs from CSV of training data
data = list( csv.reader( open(  'data.csv', 'rU' ) ) )
data = np.array(data)

# Convert data to work on GenSim
texts_matrix = data[:,3]
texts = texts_matrix.tolist()

# StopWords to ignore
stopWordsList = """a,able,about,across,after,all,almost,also,am
        ,among,an,and,any,are,as,at,be,because
        ,been,but,by,can,cannot,could,dear
        ,did,do,does,either,else,ever,every
        ,for,from,get,got,had,has,have,he,her
        ,hers,him,his,how,however,i,if,in,into
        ,is,it,its,just,least,let,like,likely,
        may,me,might,most,must,my,neither,no,nor,
        not,of,off,often,on,only,or,other,our,own,
        rather,said,say,says,she,should,since,so,
        some,than,that,the,their,them,then,there,
        these,they,this,tis,to,too,twas,us,wants,
        was,we,were,what,when,where,which,while,
        who,whom,why,will,with,would,yet,you,your""".split(",")
# Remove stopwords 
finalText = [[word for word in document.lower().split() if word not in stopWordsList]
    for document in texts]

# delete words occuring once
all_tokens = sum(finalText, [])

tok_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)

#build 2d array of doc
allDoc = [[word for word in text ] for text in texts]

# Create a dictionary 
dictionary = corpora.Dictionary(allDoc)

# get bag of words rep
features = [dictionary.doc2bow(text) for text in allDoc]

# Train LDA model
lda = models.ldamodel.LdaModel(corpus=features, id2word=dictionary, num_topics=100, update_every=0, chunksize=1, passes=2)

#get topic predictions for each doc. a csv is manually created using this. this is then manually added to main training data. 
for i, text in enumerate(features):
    topics = []
    for topic in lda[text]:
        topics.append(topic[0])
    if not entered:
        topics.append(-1)
    print topics
