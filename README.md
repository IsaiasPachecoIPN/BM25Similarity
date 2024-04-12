
# NLP Similarity Computation

This modes was created to explore some similarity methods that exists in NLP. 




## Installation

In order tu run the program, some python packages have to be installed

```bash
  pip install pandas numpy progressbar2 
```

Then run 

```bash
  python main.py
```
## Usage/Examples

```javascript
  from model import *

#Load the lemmatized corpus
lemmatized_corpus = open('./output_lematized_text.txt',"r", encoding='utf-8').read()
lemmatized_corpus = lemmatized_corpus.split(' ')

model = NLPSimilarity(lemmatized_corpus)
model.create_context_windows()
model.create_context_frequency_matrix()

#Frequency matrix normalization
model.normalize_frequency_matrix()

data = model.calculate_bm25_similarity_one_document_to_all('crecimiento', verbose=True)

#Print the first 20 most similar documents 
counter = 20
for idx,key in enumerate(data.keys()):
    print(f"[{idx}]{key}: {data[key]}")
    counter -= 1

    if counter == 0:
        break

```

