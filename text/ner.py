from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize

# Example sentence
sentence = "Barack Obama visited Paris in 2019."

# Tokenize and get POS tags
tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)

# Apply Named Entity Recognition
named_entities = ne_chunk(pos_tags)

# Print the named entities
print(named_entities)
