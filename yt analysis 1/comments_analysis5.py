# Importing necessary libraries for fetching and saving comments
from googleapiclient.discovery import build
import re
import emoji
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# API Key for accessing YouTube Data API
API_KEY = 'AIzaSyBLJ-99tf5r_Tl56Bf2hcEVaKZiYkRmSUI'

# Initializing YouTube API
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Taking input from the user and extracting the video ID
video_id = input('Enter YouTube Video URL: ')[-11:]
print("video id: " + video_id)

# Getting the channelId of the video uploader
video_response = youtube.videos().list(
    part='snippet',
    id=video_id
).execute()

# Extracting channel ID
video_snippet = video_response['items'][0]['snippet']
uploader_channel_id = video_snippet['channelId']
print("channel id: " + uploader_channel_id)

# Fetching comments
print("Fetching Comments...")
comments = []
nextPageToken = None
while len(comments) < 600:
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=100,  # You can fetch up to 100 comments per request
        pageToken=nextPageToken
    )
    response = request.execute()
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        # Check if the comment is not from the video uploader
        if comment['authorChannelId']['value'] != uploader_channel_id:
            comments.append(comment['textDisplay'])
    nextPageToken = response.get('nextPageToken')

    if not nextPageToken:
        break

# Print the first 5 comments
print(comments[:5])

# Regular expression to detect hyperlinks
hyperlink_pattern = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

threshold_ratio = 0.65
relevant_comments = []

# Filtering relevant comments
for comment_text in comments:
    comment_text = comment_text.lower().strip()
    emojis = emoji.emoji_count(comment_text)
    # Count text characters (excluding spaces)
    text_characters = len(re.sub(r'\s', '', comment_text))
    if (any(char.isalnum() for char in comment_text)) and not hyperlink_pattern.search(comment_text):
        if emojis == 0 or (text_characters / (text_characters + emojis)) > threshold_ratio:
            relevant_comments.append(comment_text)

# Print the relevant comments
print(relevant_comments[:5])

# Creating a DataFrame for the comments
comments_df = pd.DataFrame(relevant_comments, columns=['Comment'])

# Saving the DataFrame to a CSV file
comments_df.to_csv('comments.csv', index=False, encoding='utf-8')
print("Comments saved to comments.csv successfully!")

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
dataset = pd.read_csv('comments.csv')

# Data cleaning
corpus = []
for i in range(0, min(1000, len(dataset))):  # Adjust the range if needed
    review = re.sub('[^a-zA-Z]', ' ', dataset['Comment'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
# Assuming the target variable is in the last column
# Since we don't have target labels, we'll create dummy labels for the purpose of running the models
# In practice, you would have actual sentiment labels (positive, negative, neutral) for supervised learning
y = np.random.choice([0, 1], size=(len(X),))  # Dummy binary labels for demonstration

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training and evaluating models
models = {
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(kernel='linear', random_state=0),
    'Logistic Regression': LogisticRegression(random_state=0),
    'K-NN': KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=0),
    'Random Forest': RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
}

# Dictionary to store model accuracies
model_accuracies = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy
    print(f"{name} Confusion Matrix:\n{cm}")
    print(f"{name} Accuracy: {accuracy}\n")

print("Training complete.")

# Plotting the results
model_names = list(model_accuracies.keys())
accuracies = list(model_accuracies.values())

plt.figure(figsize=(12, 6))
sns.barplot(x=model_names, y=accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.show()
