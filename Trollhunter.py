import json, re
import nltk
import skleran
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Path to the JSON file
file_path = 'Dataset for Detection of Cyber-Trolls.json'
#path in colab
#file_path = '/content/Dataset for Detection of Cyber-Trolls.json'
# Reading the file
with open(file_path, 'r') as file:
    data = file.readlines()

# Convert the JSON strings to dictionaries and store them in a list
tweets = [json.loads(line) for line in data]

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Porter Stemmer (will reduce words to base form)
stemmer = PorterStemmer()

# Extracting content and labels (gets text from tweet and lable for troll or non troll)
tweets_content = [tweet['content'] for tweet in tweets]
tweets_labels = [int(tweet['annotation']['label'][0]) for tweet in tweets]

# Define a function for cleaning text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#','', text)
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokenized_text = word_tokenize(text)
    # Remove stopwords and perform stemming
    cleaned_text = [stemmer.stem(word) for word in tokenized_text if word not in stopwords.words('english')]
    
    return ' '.join(cleaned_text)

# Clean the tweets
cleaned_tweets = [clean_text(tweet) for tweet in tweets_content]

# Display the first few cleaned tweets
print(cleaned_tweets[:5], tweets_labels[:5])

#TF-IDF used here to help convert processed text into numerical fomrat
#TF stands for term frequency- amount of times words show up
#IDF- stands for Inverse document Frequency
#IDF Measures how important a term is (weights it)
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the cleaned tweets
tfidf_features = vectorizer.fit_transform(cleaned_tweets)

# Displaying the shape of the TF-IDF feature matrix
tfidf_features.shape

#split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, tweets_labels, test_size=0.2, random_state=42)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize the Naive Bayes Classifier
clf = MultinomialNB()

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Calculate and print metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

#confusion matrix plot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Cyber-Aggressive', 'Cyber-Aggressive'], 
            yticklabels=['Non Cyber-Aggressive', 'Cyber-Aggressive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
