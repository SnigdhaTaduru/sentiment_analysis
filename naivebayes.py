import pandas as pd
import numpy as np
import math
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

#-----------------------------naive bayes------------------------
def naivebayes(a,y_train,b,y_test):
    predicted_array=[]
    def expoy(xi,j):
        if(vary[j]==0):
            return 0
        y=float(math.pow((xi-meansy[j]),2))*(-1)
        h=2*vary[j]
        t=float(y/(1.0*h))
        inter=math.exp(t)
        denom=math.sqrt(2*math.pi)*math.sqrt(vary[j])
        ans=float(inter/(1.0*denom))
        return ans
    def expon(xi,j):
        if(varn[j]==0):
            return 0
        y=float(math.pow((xi-meansn[j]),2))*(-1)
        h=2*varn[j]
        t=float(y/(1.0*h))
        inter=math.exp(t)
        denom=math.sqrt(2*math.pi)*math.sqrt(varn[j])
        ans=float(inter/(1.0*denom))
        return ans
        
    county=0
    countn=0
    for i in range(cvr):
        if(y_train[i]==1):
            county+=1
        else:
            countn+=1
    print 'county ',county,' countn ',countn
    meansy=[0.0]*(cvc)
    meansn=[0.0]*(cvc)
    for i in range(cvc):
        for j in range(cvr):
            if(y_train[j]==1):
                meansy[i]+=a[j,i]
            else:
                meansn[i]+=a[j,i]
    for i in range(cvc):
        meansy[i]=float(meansy[i]/(county*1.0))
        meansn[i]=float(meansn[i]/(countn*1.0))
    vary=[0.0]*(cvc)
    varn=[0.0]*(cvc)
    for i in range(cvc):
        for j in range(cvr):
            if(y_train[j]==1):
                vary[i]+=math.pow((a[j,i]-meansy[i]),2)
            else:
                varn[i]+=math.pow((a[j,i]-meansn[i]),2)
    for i in range(cvc):
        vary[i]=float(vary[i]/((county-1)*1.0))
        varn[i]=float(varn[i]/((countn-1)*1.0))
    
    acc=0
    for i in range(cvr1):
        py=float(county/(cvr*1.0))
        pn=float(countn/(cvr*1.0))
        for j in range(cvc):
            py*=expoy(b[i,j],j)
            pn*=expon(b[i,j],j)
        #print 'py ',py,' pn ',pn
        if(py==pn or y_test[i]==2):
            pred=2
            #print 'neu'
            predicted_array.append(pred)
            acc+=1
        elif(py>pn):
            pred=1
            #print 'hi'
            predicted_array.append(pred)
            if(y_test[i]==4):
                acc+=1
        elif(pn>py):
            pred=0
            #print 'hey'
            predicted_array.append(pred)
            if(y_test[i]==0):
                acc+=1
    accuracy=float(acc/(cvr1*1.0))
    predicted_array=np.array(predicted_array)
    return predicted_array,accuracy
                
    

#-----------------------------get train and test------------------
def getTrainAndTestData():
    X=[]
    y=[]
    X=mat[0:5000,c-1]
    #x1=mat[:-2500,c-1]
    #X=np.append(X,x1)
    y=mat[0:5000,0]

    X_test=mat1[:,col1-1]
    y_test=mat1[:,0]
    
    c1=0
    c0=0
    #y1=mat[:-5000,0]
    #y=np.append(y,y1)
    for i in range(5000):
        if(y[i]==4):
            y[i]=1
            c1+=1
        else:
            y[i]=0
            c0+=1

    
    return X,y,X_test,y_test,c1,c0

#-------------------------------processTweets---------------------
def processTweets(X_train,r):
    for i in range(r):
        X_train[i]=preprocess(X_train[i])
        #print i," ",X_train[i]
    return X_train

#-------------------------------preprocess------------------------
def preprocess(tweet):
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','__URL',tweet)
    
    #Convert @username to __HANDLE
    tweet = re.sub('@[^\s]+','__HANDLE',tweet)
    
    #trim
    tweet = tweet.strip('\'"')
    
    # Repeating words like happyyyyyyyy
    rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    tweet = rpt_regex.sub(r"\1\1", tweet)
    
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    #Emoticons
    emoticons = \
    [
     ('__positive__',[ ':-)', ':)', '(:', '(-:', \
                       ':-D', ':D', 'X-D', 'XD', 'xD', \
                       '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ] ),\
     ('__negative__', [':-(', ':(', '(:', '(-:', ':,(',\
                       ':\'(', ':"(', ':((', ] ),\
    ]

    def replace_parenth(arr):
       return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]
    
    def regex_join(arr):
        return '(' + '|'.join( arr ) + ')'

    emoticons_regex = [ (repl, re.compile(regex_join(replace_parenth(regx))) ) \
            for (repl, regx) in emoticons ]
    
    for (repl, regx) in emoticons_regex :
        tweet = re.sub(regx, ' '+repl+' ', tweet)

    
    #Convert to lower case
    tweet = tweet.lower()

    #print "i ",i," tweet ",tweet
    #X[i,c-1]=tweet
    return tweet


#------------------------------start------------------------------
df=pd.read_csv('training.1600000.processed.noemoticon.csv')
df=df.sample(frac=1)
mat=df.as_matrix()
r,c=mat.shape
print "r ",r," c ",c
#for i in range(5000):
#    print mat[i,0]

df1=pd.read_csv('testdata.manual.2009.06.14.csv')
mat1=df1.as_matrix()
row1,col1=mat1.shape
print "r1 ",row1," c1 ",col1

X_train,y_train,X_test,y_test,c1,c0=getTrainAndTestData()
print "c1 ",c1," c0 ",c0
X_train=processTweets(X_train,5000)
for i in range(5000):
    arr=[]
    s=''
    st=str(X_train[i]).decode('ascii',errors="ignore")
    text=TextBlob(st)
    print text.tags
    for word,pos in text.tags:
        if((pos=='JJ' or pos=='RB' or pos=='JJR' or pos=='JJS' or pos=='RBR' or pos=='RBS') and word!='__handle'):
            arr.append(str(word))
    s=' '.join(word for word in arr)
    print 'sentence which contains only adjective and adverbs ',arr
    print 'sentece is ',s
    X_train[i]=s

print '------------------------'                
print X_train
print "over"
cv=TfidfVectorizer(min_df=0.01,max_df=0.9,max_features=500,ngram_range=(1,2),decode_error='ignore')
X_traincv=cv.fit_transform(X_train)
a=X_traincv.toarray()
cvr,cvc=a.shape
print a.shape
for i in range(cvr):
    for j in range(cvc):
        a[i,j]=a[i,j]*100
#print a
names=cv.get_feature_names()
#y_train=y_train.astype('int')
print y_train
#for i in range(5000):
#    print y_train[i]," "
#for i in range(10):
#    print a[i][0:50]
X_test=processTweets(X_test,row1)
for i in range(row1):
    arr=[]
    s=''
    st=str(X_test[i]).decode('ascii',errors="ignore")
    text=TextBlob(st)
    #print text.tags
    for word,pos in text.tags:
        if((pos=='JJ' or pos=='RB' or pos=='JJR' or pos=='JJS' or pos=='RBR' or pos=='RBS') and word!='__handle'):
            arr.append(str(word))
    s=' '.join(word for word in arr)
    #print 'sentence is ',s
    X_test[i]=s            

X_testcv=cv.transform(X_test)
b=X_testcv.toarray()
cvr1,cvc1=b.shape
print b.shape
for i in range(cvr1):
    for j in range(cvc1):
        b[i,j]=b[i,j]*100
#print b
print names
#getscore(names)
predicted_array,accu=naivebayes(a,y_train,b,y_test)
print "Accuracy is ",accu*100
