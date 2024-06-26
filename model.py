import  pickle
import  pandas as pd
import  numpy as np
import  progressbar as pb
from    collections import Counter
class NLPSimilarity:

    corpus                   = None
    vocabulary               = None
    contexts_windows         = None
    context_frequency_matrix = None
    vocabulary_idf           = None 
    bm25_one_to_all          = None

    def __init__(self, lemmatized_corpus):
        self.corpus = lemmatized_corpus

        

    def get_vocabulary(self):

        """
        Funtion to get the vocabulary of the corpus

        @param corpus:  The corpus to get the vocabulary
        @return:        The vocabulary of the corpus
        @note:          The vocabulary is stored in the file output/vocabulary.pkl and is loaded if it exists
        """

        #Check if the vocabulary is already created
        try:
            with open('./output/vocabulary.pkl', 'rb') as f:
                self.vocabulary = pickle.load(f)
        except:
            #Create the vocabulary
            self.vocabulary = set()
            for doc in self.corpus:
                if(doc != ''):
                    self.vocabulary.add(doc)
            #Save the vocabulary in the file outut/vocabulary.pkl 
            with open('./output/vocabulary.pkl', 'wb') as f:
                pickle.dump(self.vocabulary, f)

        print(f"Vocabulary size: {len(self.vocabulary)}")
        return self.vocabulary
    
    def create_context_windows(self, window_size=8):
        """"
        Create context windows of each word of the vocabulary
        @param:     window_size: the size of the context windows
        @return:    the context windows of each word of the vocabulary
        @note:      The context windows are stored in the file output/contexts_windows.pkl and are loaded if it exists
        """

        #Check if the vocabulary is already created
        if self.vocabulary is None:
            self.get_vocabulary()

        #Check if the context windows are already created
        try:
            with open('./output/contexts_windows.pkl', 'rb') as f:
                self.contexts_windows = pickle.load(f)
        except:
            #Create the context windows
            print(f"Creating context windows for each word of the vocabulary")
            bar = pb.ProgressBar(max_value=len(self.vocabulary), widgets=[pb.Bar()])
            bar.start()
            bar_count = 0
            self.contexts_windows = {}
            for word in self.vocabulary:
                self.contexts_windows[word] = []
                for i, doc in enumerate(self.corpus):
                    if(doc == word):
                        context = []
                        for j in range(i-window_size, i+window_size+1):
                            if(j >= 0 and j < len(self.corpus)):
                                context.append(self.corpus[j])
                        self.contexts_windows[word].extend(context)
                bar_count += 1
                bar.update(bar_count) 
            bar.finish()

            #Save the context windows in the file outut/contexts_windows.pkl 
            with open('./output/contexts_windows.pkl', 'wb') as f:
                pickle.dump(self.contexts_windows, f)    
        
        print(f"Context windows created for each word of the vocabulary")

    def create_context_frequency_matrix(self, verbose=False):
        """
        Create the context frequency matrix
        @param verbose: a boolean to print the head of the context frequency matrix
        return:         the context frequency matrix as a pandas dataframe
        @note:          The context frequency matrix is stored in the file output/context_frequency_matrix.pkl and is loaded if it exists
        """

        #Check if the matrix is already created
        try:
            with open('./output/context_frequency_matrix.pkl', 'rb') as f:
                self.context_frequency_matrix = pickle.load(f)
        except:
            #Check if the vocabulary is already created
            if self.vocabulary is None:
                self.get_vocabulary()

            #Check if the context windows are already created
            if self.contexts_windows is None:
                self.create_context_windows()

            vocabulary_size = len(self.vocabulary)
            #store the result in a pandas dataframe
            df = pd.DataFrame(np.zeros((vocabulary_size, vocabulary_size)), index=list(self.vocabulary), columns=list(self.vocabulary), dtype=np.float64)

            print(f"Creating context frequency matrix")
            bar = pb.ProgressBar(max_value=vocabulary_size)
            bar_count = 0
            bar.start()
            for cxt in self.vocabulary:
                context_window_counts = Counter(self.contexts_windows[cxt])
                #print(f"Context window counts for {cxt}: {context_window_counts}")
                for word in context_window_counts.keys():
                    if word == '':
                        continue
                    df.at[cxt, word] = context_window_counts[word]
                bar_count += 1
                bar.update(bar_count)
            bar.finish()

            #Save the context frequency matrix in the file outut/context_frequency_matrix.pkl
            with open('./output/context_frequency_matrix.pkl', 'wb') as f:
                pickle.dump(df, f)

            #Print the dataframe head    
            self.context_frequency_matrix = df


        if verbose:
            print(self.context_frequency_matrix.head())
            print(f"Context frequency matrix shape: {self.context_frequency_matrix.shape}")

        return self.context_frequency_matrix

    def get_document_frequency_matrix(self, document):
        
        """
        Funtion to get the document frequency matrix
        @param document     The document to get the frequency matrix
        @return:            The frequency matrix of the document
        """

        #Check if the frequency matrix is already created
        if self.context_frequency_matrix is None:
            self.create_context_frequency_matrix()
        else:
            return self.context_frequency_matrix[document]

    def get_max_frequency_word(self, document):
        """
        Get the word with the maximum frequency in a document
        @param document     The document to get the word
        @return:            The word with the maximum frequency in the document
        """

        #Check if the frequency matrix is already created
        if self.context_frequency_matrix is None:
            self.create_context_frequency_matrix()
        else:
            return self.context_frequency_matrix[document].idxmax()

    def get_document_word_frequency(self, document, word):

        """
        Function to get the frequency of a word in a document
        @param document:    The document to get the frequency
        @param word:        The word to get the frequency
        return:             The frequency of the word in the document
        """
            
        #Check if the frequency matrix is already created
        if self.context_frequency_matrix is None:
            self.create_context_frequency_matrix()
        else:
            return self.context_frequency_matrix.at[document, word]

    def get_word_context_window_array(self, word):
        """
        Get the context windows of a word
        @param word:    The word to get the context windows
        @return:        The context windows of the word
        """

        #Check if the context windows are already created
        if self.contexts_windows is None:
            self.create_context_windows()

        return self.contexts_windows[word]

    def normalize_frequency_matrix(self, use_l2_norm=True):
        
        """
        Normalize the frequency matrix by normalizing the context frequency array of every document
        @param use_l2_norm: A boolean to use the l2 norm
        return:             The normalized frequency matrix
        """

        if self.vocabulary is None:
            self.get_vocabulary()

        if self.context_frequency_matrix is None:
            self.create_context_frequency_matrix()
        
        print("Normalizing frequency matrix")

        for word in self.vocabulary:
            ctx_freq_arr = self.context_frequency_matrix.loc[word].to_numpy()
            ctx_freq_arr_norm = np.linalg.norm(ctx_freq_arr)
            if use_l2_norm:
                self.context_frequency_matrix.loc[word] = ctx_freq_arr/np.linalg.norm(ctx_freq_arr)
            else:
                self.context_frequency_matrix.loc[word] = ctx_freq_arr/ctx_freq_arr_norm
        
        print(f"Frequency matrix normalized")
        return self.context_frequency_matrix
        
    def calculate_document_idf(self, word):
        """
        Get the inverse document frequency of a word of the vocabulary
        @param word:    The word to get the idf
        @return:        The idf of the word
        """

        #Check if the vocabulary is already created
        if self.vocabulary is None:
            self.get_vocabulary()

        #check if the contexts windows are already created
        if self.contexts_windows is None:
            self.create_context_windows()

        #Number of documents
        idf = 0
        N = len(self.contexts_windows)
        df = 0

        for ctx in self.contexts_windows.keys():
            if word in self.contexts_windows[ctx]:
                df += 1

        idf = np.log((N+1)/df)

        return idf

    def get_vocabulary_documents_idf(self):
        """
        Get the inverse document frequency of each word of the vocabulary and generate a dictionary
        @return:    The idf of each word
        @note:      The idf of each word is stored in the file output/vocabulary_idf.pkl and is loaded if it exists
        """
        print("Getting the idf of each word of the vocabulary")

        #Check if the vocabulary is already created
        if self.vocabulary is None:
            self.get_vocabulary()

        #Check if the vocabulary is already created
        try:
            with open('./output/vocabulary_idf.pkl', 'rb') as f:
                self.vocabulary_idf = pickle.load(f)
        except:
            self.vocabulary_idf = {}

            bar = pb.ProgressBar(max_value=len(self.vocabulary), widgets=[pb.Bar()])
            bar.start()
            bar_count = 0
            for word in self.vocabulary:
                self.vocabulary_idf[word] = self.calculate_document_idf(word)
                bar_count += 1
                bar.update(bar_count)
            bar.finish()

            #Save the vocabulary idf in the file outut/vocabulary_idf.pkl
            with open('./output/vocabulary_idf.pkl', 'wb') as f:
                pickle.dump(self.vocabulary_idf, f)

        print(f"Idf of each word of the vocabulary loaded")
        return self.vocabulary_idf
    

    def get_document_idf(self, word):
        """
        Get the inverse document frequency of a document
        @param word:    The word to get the idf
        @return:        The idf of the word
        """

        #Check if the vocabulary idf is already created
        if self.vocabulary_idf is None:
            self.get_vocabulary_documents_idf()

        return self.vocabulary_idf[word]
    
    def get_lower_idf_word(self):
        """
        Get the word with the lower idf
        return: the word with the lower idf
        """

        #Check if the vocabulary idf is already created
        if self.vocabulary_idf is None:
            self.get_vocabulary_documents_idf()

        return min(self.vocabulary_idf, key=self.vocabulary_idf.get)
    
    def get_higher_idf_word(self):
        """
        Get the word with the higher idf
        return: the word with the higher idf
        """

        #Check if the vocabulary idf is already created
        if self.vocabulary_idf is None:
            self.get_vocabulary_documents_idf()

        return max(self.vocabulary_idf, key=self.vocabulary_idf.get)

    def plot_vocabularty_idf(self, limit=100):
        """
        Plot the idf of the vocabulary using matplotlib
        """

        #Check if the vocabulary idf is already created
        if self.vocabulary_idf is None:
            self.get_vocabulary_documents_idf()

        import matplotlib.pyplot as plt


        plt.figure(figsize=(20,10))
        plt.bar(self.vocabulary_idf.keys()[:limit], self.vocabulary_idf.values()[:limit])
        plt.xlabel('Words')
        plt.ylabel('IDF')
        plt.title('Inverse Document Frequency of the Vocabulary')
        plt.show()
    
    def calculate_document_BM25_array(self,document,d_len, avdl, k=1.2, b=0.75, verbose=False):
        """" 
        Function to calculate the BM25 of a document
        @param document:    The document to calculate the BM25
        @param k:           The k parameter of the BM25
        @param b:           The b parameter of the BM25
        @return:            The BM25 array of the document
        """
        if self.context_frequency_matrix is None:
            self.create_context_frequency_matrix()

        document_freq_array = self.get_document_frequency_matrix(document)
        
        bm25_denominator = document_freq_array + k * (1 - b + b * d_len / avdl)
        bm25_constant = (k + 1) / bm25_denominator
        bm25 = bm25_constant * document_freq_array

        return bm25


    def calculate_document_word_BM25(self,document, word,d_len, avdl, k=1.5, b=0.75, verbose=False):
        """
        Function to calculate the BM25 of certain word in a document
        @param document:    The document to calculate the BM25
        @param word:        The word to calculate the BM25
        @param k:           The k parameter of the BM25
        @param b:           The b parameter of the BM25
        @return:            The BM25 of the word in the document
        """

        word_ctx_freq = self.get_document_word_frequency(document,word)
        
        if word_ctx_freq == 0:
            return 0
        
        bm25  = 0

        if verbose:
            print(f"*******************************************************")
            print(f"Document: {document}")
            print(f"Word: {word}")
            print(f"Frequency: {word_ctx_freq}")
            print(f"Document length: {d_len}")
            print(f"Average document length: {avdl}")
            print(f"*******************************************************")


        bm25 = ((k+1)*word_ctx_freq)/( word_ctx_freq + k*(1-b+b*(d_len/avdl)))
    
        return bm25
    
    def calculate_document_lenght(self, document):
        """
        Function to calculate the length of a document
        @param document:    The document to calculate the length
        @return:            The length of the document
        """

        #Check if the context windows are already created
        if self.contexts_windows is None:
            self.create_context_windows()

        return np.abs(len(self.contexts_windows[document]))

    def calculate_documents_length_average(self):
        """
        Function to calculate the average length of the documents
        @return: the average length of the documents
        """

        #Check if the context windows are already created
        if self.contexts_windows is None:
            self.create_context_windows()

        total_length = 0

        for word in self.contexts_windows.keys():
            total_length += len(self.contexts_windows[word])

        return total_length/len(self.contexts_windows)

    def calculate_bm25_smilarity(self, document_1, document_2, verbose=False):
        """
        Function to calculate the BM25 similarity between two documents
        @param document1:   The first document
        @param document2:   The second document
        @return:            The BM25 similarity between the two documents
        """

        bm25_sim = 0

        #Check if the idf of the vocabulary is already created
        if self.vocabulary_idf is None:
            self.get_vocabulary_documents_idf()

        d1_len = self.calculate_document_lenght(document_1)
        d2_len = self.calculate_document_lenght(document_2)
        avdl = self.calculate_documents_length_average()

        xi = self.calculate_document_BM25_array(document_1,d1_len, avdl)
        yi = self.calculate_document_BM25_array(document_2,d2_len, avdl)

        # xi = xi/xi.sum()
        # yi = yi/yi.sum()

        xi = xi / (np.linalg.norm(xi))
        yi = yi / (np.linalg.norm(yi))

        idf_values = np.array([self.vocabulary_idf[elem] for elem in self.context_frequency_matrix.columns])

        bm25_sim =  np.dot(idf_values * xi, yi)

        if verbose:
            print(f"*******************************************************")
            print(f"Average document length: {avdl}")
            print(f"BM25 similarity: {bm25_sim}")
            print(f"*******************************************************")

        return bm25_sim
    
    def calculate_bm25_similarity_one_document_to_all(self, document, override=False, verbose=False, load_external_data=False, external_data=None):
        """
        Function to get the BM25 similarity of a document to all the documents
        return a sorted object with the BM25 similarity document:doc
        """

        print(f"Calculating BM25 similarity of the document {document} to all the documents")

        #Check if there is a previous calculation
        try:
            if override:
                raise Exception("Override")
            
            if load_external_data:
                with open(external_data, 'rb') as f:
                    self.bm25_one_to_all = pickle.load(f)
            else:
                with open('./output/bm25_one_to_all.pkl', 'rb') as f:
                    self.bm25_one_to_all = pickle.load(f)
        except:
            #Check if the vocabulary is already created
            if self.vocabulary is None:
                self.get_vocabulary()


            bm25_sim = {}

            bar = pb.ProgressBar(max_value=len(self.vocabulary), redirect_stdout=True)
            bar.start()
            bar_count = 0

            d1_len = self.calculate_document_lenght(document)
            avdl = self.calculate_documents_length_average()
            
            xi = self.calculate_document_BM25_array(document,d1_len, avdl)

            xi = xi / (np.linalg.norm(xi))

            idf_values = np.array([self.vocabulary_idf[elem] for elem in self.context_frequency_matrix.columns])

            for doc in self.vocabulary:
                yi = self.calculate_document_BM25_array(doc,self.calculate_document_lenght(doc), avdl)
                yi = yi / (np.linalg.norm(yi))
                bm25_sim[f"{document}:{doc}"] = np.dot(idf_values * xi, yi)

                bar_count += 1
                
                if verbose:
                    #print(f"BM25 similarity {document}:{doc}: {bm25_sim[f'{document}:{doc}']}")
                    bar.update(bar_count)
                else:
                    bar.update(bar_count)
            bar.finish()

            bm25_sim = dict(sorted(bm25_sim.items(), key=lambda item: item[1], reverse=True))

            self.bm25_one_to_all = bm25_sim

            #Save the bm25 similarity in the file outut/bm25_one_to_all.pkl
            with open('./output/bm25_one_to_all.pkl', 'wb') as f:
                pickle.dump(bm25_sim, f)

        return self.bm25_one_to_all
    
    def calculate_cosine_similarity(self, document_1, document_2, verbose=False):
        """
        Function to calculate the cosine similarity between two documents
        param document1: the first document
        param document2: the second document
        return: the cosine similarity between the two documents
        """

        #Check if the frequency matrix is already created
        if self.context_frequency_matrix is None:
            self.create_context_frequency_matrix()

        #Get the frequency matrix of the documents
        d1 = self.context_frequency_matrix.loc[document_1].to_numpy()
        d2 = self.context_frequency_matrix.loc[document_2].to_numpy()

        #Calculate the cosine similarity
        cosine_sim = np.dot(d1,d2)/(np.linalg.norm(d1)*np.linalg.norm(d2))

        if verbose:
            print(f"*******************************************************")
            print(f"Document 1: {document_1}")
            print(f"Document 2: {document_2}")
            print(f"Cosine similarity: {cosine_sim}")
            print(f"*******************************************************")

        return cosine_sim
    
    def calculate_cosine_similarity_one_to_all(self, document1, verbose=False):
        """
        Function to calculate the cosine similarity of a document to all the documents
        """

        documents_cosine_sim = {}

        for doc in self.vocabulary:
            documents_cosine_sim[f"{document1}:{doc}"] = self.calculate_cosine_similarity(document1, doc)

        documents_cosine_sim = dict(sorted(documents_cosine_sim.items(), key=lambda item: item[1], reverse=True))

        return documents_cosine_sim 