from model import *

#Load the lemmatized corpus
lemmatized_corpus = open('./output_lematized_text.txt',"r", encoding='utf-8').read()
lemmatized_corpus = lemmatized_corpus.split(' ')

model = SimilarityBM25(lemmatized_corpus)

#Print the vocabulary
# print(model.get_vocabulary())

#print(model.get_vocabulary())
model.create_context_frequency_matrix()

#print(model.get_document_frequency_matrix('crecimiento'))
print(f"Document frequency {(model.get_document_word_frequency('crecimiento', 'crecimiento'))}")

#Frequency matrix normalization
model.normalize_frequency_matrix()

print(model.context_frequency_matrix.head())

#print(model.calculate_bm25_smilarity(document_1='crecimiento',document_2='crecimiento', verbose=False))
print(model.calculate_bm25_similarity_one_document_to_all('crecimiento', verbose=True))
