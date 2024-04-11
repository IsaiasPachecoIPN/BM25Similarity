from model import *
import pandas as pd

#Load the lemmatized corpus
lemmatized_corpus = open('./output_lematized_text.txt',"r", encoding='utf-8').read()
lemmatized_corpus = lemmatized_corpus.split(' ')

model = NLPSimilarity(lemmatized_corpus)

#Print the vocabulary
# print(model.get_vocabulary())

#print(model.get_vocabulary())

model.create_context_windows()
model.create_context_frequency_matrix()

#print(model.get_document_frequency_matrix('crecimiento'))

# print(f"Max value of document': {model.get_max_frequency_word('crecimiento')}")
# print(f"Document frequency {(model.get_document_word_frequency('crecimiento', 'crecimiento'))}")

# ctx_freq_arr = model.context_frequency_matrix.loc['crecimiento'].to_numpy()
# ctx_freq_arr_norm = np.linalg.norm(ctx_freq_arr)
# print(f"Norm of the context frequency array: {ctx_freq_arr_norm}")

#Frequency matrix normalization
model.normalize_frequency_matrix()

model.calculate_bm25_smilarity(document_1='crecimiento',document_2='crecimiento', verbose=True)

# print(f"Document frequency {(model.get_document_word_frequency('crecimiento', 'crecer'))}")
# print(model.context_frequency_matrix.head())

# model.calculate_bm25_smilarity(document_1='crecimiento',document_2='ambicioso', verbose=True)
# model.calculate_cosine_similarity(document_1='crecimiento',document_2='crecimiento', verbose=True)

data = model.calculate_bm25_similarity_one_document_to_all('crecimiento', verbose=True, load_external_data=True, override=True)

counter = 10
for idx,key in enumerate(data.keys()):
    print(f"[{idx}]{key}: {data[key]}")
    counter -= 1

    if counter == 0:
        break

# print("********************************************************************************")

# data = model.calculate_cosine_similarity_one_to_all('crecimiento', verbose=True)

# counter = 100
# for key in data.keys():
#     print(f"{key}: {data[key]}")
#     counter -= 1

#     if counter == 0:
#         break