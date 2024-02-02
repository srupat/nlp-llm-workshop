import string
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, RegexpParser, ne_chunk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download('punkt') # to tokenize 
nltk.download('stopwords') # to remove stopwords
nltk.download('vader_lexicon') # assigns sentiment scores to words
nltk.download('maxent_ne_chunker') # for named entity recognition
nltk.download('words') 
nltk.download('wordnet') # lexical db for lemmatization
nltk.download('averaged_perceptron_tagger') # part-of-speech tagging

# most of the text on the net is of the encoding 'utf-8'
text = open('textRead.txt', encoding='utf-8').read()
# print(text)

# convert to lower case
lower_case = text.lower()
# print(lower_case)

# remove all punctuations
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
# print(cleaned_text)

# split the sentence into words that are then stored in a list
tokenized_words = word_tokenize(cleaned_text, "english")
# print(tokenized_words)

# final words is a list of words that do not contain stop words
final_words = []
stop_words = set(stopwords.words('english'))
for word in tokenized_words:
    if word not in stop_words:
        final_words.append(word)

# print(final_words)

# apply stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in final_words]

# print(stemmed_words)

# apply lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in final_words]

# print(lemmatized_words)

# apply Parts of Speech Tagging
pos_tags = pos_tag(final_words)

# print(len(pos_tags))

# apply Dependency Parsing
grammar = r'NP: {<DT>?<JJ>*<NN>}' # noun phrase (det + adj + noun)
chunk_parser = RegexpParser(grammar)
chunks = chunk_parser.parse(pos_tags)
# print(chunks)

# apply Named Entity Recognition
named_entities = ne_chunk(pos_tags)

print(named_entities)

# emotion extraction using a predefined list of emotions
emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        if word in final_words:
            emotion_list.append(emotion)

# counter using collections library (to count emotions)
emotion_counter = Counter(emotion_list)

# perform sentiment analysis
def sentiment_analyze(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        return "Negative sentiment"
    elif pos > neg:
        return "Positive sentiment"
    else:
        return "Neutral sentiment"

sentiment_result = sentiment_analyze(cleaned_text)
print(sentiment_result)

# making a subplot
fig, ax1 = plt.subplots()

# bar graph generation with keys(emotions) on x-axis and values(count) on y-axis
ax1.bar(emotion_counter.keys(), emotion_counter.values())

# slanting emotions below x-axis
fig.autofmt_xdate()

# save to graph.png file
plt.savefig('graph.png')

# show graph on screen
plt.show()
