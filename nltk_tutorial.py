import nltk

# Download required NLTK data (only needed once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# A sample sentence
sentence = "Developed REST APIs using Python, Django, PostgreSQL, Docker and Kubernetes."

# Step 1: Tokenize
tokens = word_tokenize(sentence)
print("Tokens:", tokens)

# Step 2: Load stopwords
stop_words = set(stopwords.words('english'))
print("Sample stopwords:", list(stop_words)[:10])

# Step 3: Remove stopwords and punctuation
filtered_tokens = [
    word.lower()
    for word in tokens
    if word.lower() not in stop_words and word.isalpha()
]

print("Filtered tokens:", filtered_tokens)