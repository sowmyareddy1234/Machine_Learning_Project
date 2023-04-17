 #Create a ml model and create a webapp to predict if a message entered is spam or not spam

# create the same model using 4 different methods 
# 1. svm 
# 2. svm + pipeline
# 3. naive bayes 
# 4. naive bayes + pipeline 

# we have to use the model which gives best accuracy

import pandas as pd

df = pd.read_table("https://raw.githubusercontent.com/arib168/data/main/spam.tsv")
print(df)

print(df.info())

# x = df.iloc[:,1].values
x = df['message'].values #for numerical data, input x should be in 2 dimensions, for text data, it is 1 dimension only

y = df['label'].values
# y = df.iloc[:,0].values

print(df['message'][5567])

print(df['label'].value_counts())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

print(x_train.shape)

print(x_test.shape)

# tokenization - splitting the sentence into words
# vectorization - after splitting, counting how many times each word has been repeated 

#apply the feature extraction technique using the count vectorizer/bag of words

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english')
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)

#METHOD 1
from sklearn.svm import SVC 
model1 = SVC()

model1.fit(x_train_vect,y_train)  #we need x and y for fitting the model

y_pred1 = model1.predict(x_test_vect)
print(y_pred1) #predicted value

y_test  #actual value

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred1,y_test))

# METHOD 2
# using the sklearn pipeline for svc
# pipeline is used to combine two estimators/two different processes together 
# The two estimators combined here are SVC and CountVectorizer(for the pipeline)
#      Why should we use the pipeline?
# Ans:- it removes the need for us to do fit and transform individually

# fit and transform both are done simultaneously if we use the pipeline to make things easy 

from sklearn.pipeline import make_pipeline
model2 = make_pipeline(CountVectorizer(),SVC())
model2.fit(x_train,y_train)

y_pred2 = model2.predict(x_test)
print(y_pred2)

print(y_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred2,y_test))

# METHOD 3 Using Naive Bayes classifier 
from sklearn.naive_bayes import MultinomialNB
model3 = MultinomialNB()

model3.fit(x_train_vect,y_train)

y_pred3 = model3.predict(x_test_vect)
print(y_pred3)

print(y_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred3,y_test))

#METHOD 4 - MultinomialNB pipeline

from sklearn.pipeline import make_pipeline  
model4 = make_pipeline(CountVectorizer(),MultinomialNB())
model4.fit(x_train,y_train)
y_pred4 = model4.predict(x_test)
print(y_pred4)

print(y_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred4,y_test))

# SVC Model Accuracy                  - 0.9813352476669059
# SVC pipeline Accuracy               - 0.9834888729361091
# MultinomialNB Model Accuracy        - 0.9863603732950467
# MultinomialNB pipeline Accuracy     - 0.9885139985642498

# model persistance (pickle-multilinear regression )
# serializationa and deserialization steps

# we are going to persist with the best model out of the 4 models created (i.e - use the model with the highest accuracy)

#serialization
import joblib 
joblib.dump(model4,'spam-ham') # a file is created 

#desrialization
import joblib 
text_model = joblib.load('spam-ham')

text_model.predict(["free tickets sold"])   #model prediction of the output

# CREATING THE WEB APPLICATION USING STREAMLIT FOR THE SPAM HAM PREDICTION 


import streamlit as st

%%writefile demo.py 
import streamlit as st 
import joblib 

st.title("SPAM HAM CLASSIFIER")   #title for the webapp
text_model = joblib.load('/content/spam-ham') #loading the joblib model to use for predicting the output 
ip = st.text_input("Enter the message :")     #Input message 
op = text_model.predict([ip])                 # use the model for predicting the output
if st.button('PREDICT'):                      # create a button called as predict, and if that button is clicked, then display the output 
  st.title(op[0])  #print the output 

!streamlit run demo.py &npx localtunnel --port 8501
