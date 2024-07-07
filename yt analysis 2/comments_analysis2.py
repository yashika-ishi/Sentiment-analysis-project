# -----------------------Sentiment Analysis------------------------------
# -----------------1. Machine Learning-Based Approach-----------------------------

#importing liberaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#---------------------Data Cleaning-----------------------------------------------------
#cleaning the dataset(removing stopwords,stemming).
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ',
    dataset['Review'][i])
    review = review.lower()
    review = review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
review = [ps.stem(word) for word in review if
not word in set(all_stopwords)]
review = ' '.join(review)
corpus.append(review)
#Creating the Bag of Words model
from sklearn.feature_extraction.text import
CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 0)

#Training the Naive Bayes, SVM , K-NN model , Decision Tree model , Random Forest model on the dataset
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear',random_state = 0)
classifier.fit(X_train, y_train)


from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state =0)
classifier.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors= 5, metric = 'minkowski',p = 2)
classifier.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion ='entropy', random_state= 0)
classifier.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Making the Confusion Matrix of all the classificaion models
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# # If want to add wordcloud (first install also)
# codecomments_
# positive=comments[comments['pola
# rity']==1]
# !pip install wordcloud
# total_comments=''.
# join(comments_positive['comment_text'])
# wordcloud=WordCloud(width=1000,height=50
# 0,stopwords=stopwords).generate(total_comme
# nts)
# plt.figure(figsize=(15,5))
# plt.imshow(wordcloud)
# plt.axis('off')
