'''
George Lyu
gml8@rice.edu
03/14/2023

CAAM 470
Miniproject 2
'''

import numpy as np
import pandas as pd
import string
from time import time
import matplotlib.pyplot as plt

# Length of Wordle words
WORD_LEN = 5

# Desired number of disjoint words
TARGET_CLIQUE_SIZE = 5

# Read in dataset
words = [line.rstrip() for line in open('Miniproject 2/WordleWords.txt')]

# Create letter cover matrix
letters = [chr for chr in string.ascii_lowercase]
lst = [[(chr in word) for chr in letters]
                for word in words]
letter_cover = pd.DataFrame(lst,
                index=words,
                columns=letters,
                dtype=bool)

# Print summary information
print("# words                         " + str(len(letter_cover.index)))
print("# redundant anagrams            " + str(len(letter_cover.index) - len(letter_cover.drop_duplicates())))
print("# words with duplicate letters  " + str(sum(letter_cover.sum(axis=1) < WORD_LEN)))

# Drop redundant anagrams and duplicate-letters words
letter_cover = letter_cover[letter_cover.sum(axis=1) == WORD_LEN]
letter_cover = letter_cover.drop_duplicates()
print("# words after cleaning          " + str(len(letter_cover.index)))

# Create adjacency matrix
letter_cover_arr = np.asarray(letter_cover)
adj_mat_arr = np.matmul(letter_cover_arr, np.transpose(letter_cover_arr))
adj_mat_arr = np.ones(adj_mat_arr.shape) - adj_mat_arr
adj_mat_df = pd.DataFrame(adj_mat_arr,
                        dtype=bool,
                        index=letter_cover.index.tolist(),
                        columns=letter_cover.index.tolist())

class WordleCoverFinder:
    '''
    This class contains methods for finding a 5-clique of disjoint Wordle words.
    '''
    adj_mat = None          # Adjacency matrix of graph where nodes are words (without anagrams or duplicate-letter words). Pandas boolean dataframe with identical row/column indices as the words.
    timeout = 60 * 5        # Maximum number of seconds to run algorithm before timing out and returning the current list of words
    iters = 0               # Counter for the number of algorithm iterations
    start_timestamp = 0     # Timestamp for when the algorithm iterations started
    timestamps = []         # List of timestamps for when a new largest clique is discovered
    largest_clique = []     # List of the largest cliques discovered
    
    def __init__(self, adj_mat):
        '''
        Initializes this WordleCoverFinder object.
        INPUT
            adj_mat; Adjacency matrix of graph where nodes are words (without anagrams or duplicate-letter words). Pandas boolean dataframe with identical row/column indices as the words.
        '''
        self.adj_mat = adj_mat

    def solve_wordle(self):
        '''
        Solves the Wordle 5-clique problem and initializes class properties for recording algorithm performance metrics.
        '''
        self.iters = 0                  # initialize zero iterations
        self.timestamps = [0]           # initialize first timestamp
        self.largest_clique = [0]       # initialize first found depth
        self.start_timestamp = time()   # record current timestamp as starting time
        return self.find_clique(self.adj_mat, TARGET_CLIQUE_SIZE) # call find_clique iterations

    def find_clique(self, adj_mat, target_clique_size):
        '''
        Iteratively finds cliques within the given adjacency matrix of the requested size
        INPUT
            adj_mat; Adjacency matrix of graph where nodes are words (without anagrams or duplicate-letter words). Pandas boolean dataframe with identical row/column indices as the words.
            target_clique_size; size of the clique to find
        RETURNS
            list of words that form a clique of target_clique_size size within the graph represented by adj_mat, if one exists
            None, if no such clique of target_clique_size size exists within the graph
            empty list [], if the runtime exceeds the timeout this iteration
        '''
        self.iters += 1

        # If runtime exceeds timeout, return empty list
        if time() - self.start_timestamp > self.timeout:
            self.timestamps.append(time() - self.start_timestamp)
            self.largest_clique.append(self.largest_clique[-1])
            return []
        
        # If the target clique size is 1, then return any of the nodes/words in the graph.
        if target_clique_size == 1:
            self.timestamps.append(time() - self.start_timestamp)
            self.largest_clique.append(TARGET_CLIQUE_SIZE)
            return [adj_mat.index[0]]

        # Foreach node/word in the graph,  test if the word/node is a valid member of a clique.
        for candidate_word in adj_mat.index:

            # Get list of nodes/words adjacent to the candidate word
            adj_vec = adj_mat[candidate_word]

            # If the candidate word has any neighbors, check recursively if there is a target_clique_size-1 size clique among the subgraph of only these neighbors
            if adj_vec.sum() > 0:
                
                # If a new largest clique has been discovered, record it
                if TARGET_CLIQUE_SIZE - target_clique_size + 1 > self.largest_clique[-1]:
                    self.timestamps.append(time() - self.start_timestamp)
                    self.largest_clique.append(TARGET_CLIQUE_SIZE - target_clique_size + 1)

                # Create the subgraph of only the candidate word's neighbors
                sub_adj_mat = adj_mat.loc[adj_vec, adj_vec]

                # Check for any cliques within the subgraph
                clique = self.find_clique(sub_adj_mat, target_clique_size=target_clique_size-1)

                # If a clique exists within the subgraph, add candidate_word to the clique and return
                if not clique is None:
                    clique.append(candidate_word)
                    return clique
                
                # If no clique exists within the subgraph, then delete candidate_word from the neighbors of all other candidate words, since candidate_word cannot be in a clique of the subgraph of any other candidate word's neighbors
                else:
                    adj_mat.loc[candidate_word, :] = False
                    adj_mat.loc[:, candidate_word] = False

        # If no clique exists within the subgraph of the neighbors of any of the nodes/words in 
        return None
    
    def sort_words(self, rating):
        '''
        Sorts the row/col indices of the adjacency matrix by the given rating, in ascending order
        rating; a function that accepts the indices of the adjacency matrix and returns some scalar value, for which the index will be ordered by
        '''
        sorted_index = sorted(self.adj_mat.index, key=rating)
        self.adj_mat = self.adj_mat.loc[sorted_index, sorted_index]

    def print_performance(self):
        '''
        Prints performance metric information
        '''
        print('Time         ' + str(self.timestamps[-1]))
        print('Iterations   ' + str(self.iters))
        print('Iters / sec  ' + str(self.iters / self.timestamps[-1]))

    def plot_clique_discovery(self, label):
        '''
        Plots the largest discovered clique size against the runtime.
        INPUT
            label; string label for this time series
        '''
        plt.plot(wdl.timestamps, wdl.largest_clique, label=label)

# Letter-prevalence ascending
label = "Prevalence ascending"
print("----- Solving with preprocess: " + label + " -----")
prevalence_series = np.matmul(letter_cover, letter_cover.sum(axis=0))
order_rating = lambda word: prevalence_series[word]
wdl = WordleCoverFinder(adj_mat_df)
wdl.sort_words(order_rating)
print(wdl.solve_wordle())
wdl.print_performance()
wdl.plot_clique_discovery(label)

# Letter-prevalence descending
label = "Prevalence descending"
print("----- Solving with preprocess: " + label + " -----")
order_rating = lambda word: -prevalence_series[word]
wdl = WordleCoverFinder(adj_mat_df)
wdl.sort_words(order_rating)
print(wdl.solve_wordle())
wdl.print_performance()
wdl.plot_clique_discovery(label)

# Degree ascending
label = "Degree ascending"
print("----- Solving with preprocess: " + label + " -----")
degree_series = adj_mat_df.sum(axis=0)
order_rating = lambda word: degree_series[word]
wdl = WordleCoverFinder(adj_mat_df)
wdl.sort_words(order_rating)
print(wdl.solve_wordle())
wdl.print_performance()
wdl.plot_clique_discovery(label)

# Degree descending
label = "Degree descending"
print("----- Solving with preprocess: " + label + " -----")
order_rating = lambda word: -degree_series[word]
wdl = WordleCoverFinder(adj_mat_df)
wdl.sort_words(order_rating)
print(wdl.solve_wordle())
wdl.print_performance()
wdl.plot_clique_discovery(label)

# Alphabetical
label = "Alphabetical"
print("----- Solving with preprocess: " + label + " -----")
wdl = WordleCoverFinder(adj_mat_df)
wdl.sort_words(None)
print(wdl.solve_wordle())
wdl.print_performance()
wdl.plot_clique_discovery(label)

# Plot algorithm performance
plt.yticks(range(TARGET_CLIQUE_SIZE + 1))
plt.xscale('log')
plt.title('Find Clique Algorithm after Presorting')
plt.xlabel('Runtime (sec)')
plt.ylabel('Largest discovered clique size')
plt.legend(loc='lower right')
plt.show()