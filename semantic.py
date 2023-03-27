# Run the code from L3T12 Notes
# This will compare the similarity results using two english models: 'en_core_web_md' and 'en_core_web_sm'

import spacy
nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

more_tokens = nlp('plane bus chocolate crisps')
for token1 in more_tokens:
    for token2 in more_tokens:
        print(token1.text, token2.text, token1.similarity(token2))


'''MY THOUGHTS:
As expected you can see that apple and banana had relatively high similarity because they are both fruit. 
However, what was more interesting, is that compared with apple, banana scored a higher similarity 
with monkey - most likely because they are known to eat bananas. Unsurprisingly, the max result of '1' is 
received where there is a word match, for example 'cat cat'.

I also ran the code with other items to see that the likes of foods (chocolate and crisps) and modes of transport (bus and plane)
are similar. Interestingly, the similarity of bus and chocolate returned a negative result "bus chocolate -0.0003212330921087414"
I assume this is because the vectors for the two tokens have no or almost no relation/similarity to one another.
'''

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

'''
Runing the example file with the simpler language model "en_core_web_sm" results in the following warning:

UserWarning: [W007] The model you're using has no word vectors loaded, 
so the result of the Doc.similarity method will be based on the tagger, 
parser and NER, which may not give useful similarity judgements. 
This may happen if you're using one of the small models, e.g. `en_core_web_sm`, 
which don't ship with word vectors and only use context-sensitive tensors. 
You can always add your own word vectors, or use one of the larger models instead if available.

I also noted that it resulted in different similarity results but then 'en_core_web_sm' does not
include word vectors so may not be as effective for comparing the similarity of different words.

REF - https://spacy.io/models/en#en_core_web_md
REF - https://spacy.io/models/en#en_core_web_sm
REF - https://botflo.com/courses/intro-to-spacy/lessons/what-are-spacy-models/
'''
