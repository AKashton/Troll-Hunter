import json, re
import nltk
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
cleaned_tweets[:5], tweets_labels[:5]
