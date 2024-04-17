import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from config import train_dataset_path

def split_into_chunks(text, max_words_per_chunk=5):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Initialize variables
    chunks = []
    current_word_count = 0
    
    # Iterate through each sentence
    big_sentence = ''
    for idx, sentence in enumerate(sentences):
        # Calculate the word count of the current sentence
        word_count = len(sentence.split())
        
        # Check if adding the current sentence to the chunk exceeds the max word limit
        if current_word_count + word_count <= max_words_per_chunk:
            big_sentence = big_sentence + sentence
            current_word_count += word_count
        else:
            # Add the big sentence to the list of chunks
            chunks.append([big_sentence])
            # Start a new big sentence with the current sentence
            big_sentence = sentence
            current_word_count = word_count
            
        if idx == len(sentences)-1:
            chunks.append([big_sentence])
   
    
    return chunks



def run():
    text = 'I am testing the. I am testing. I am testing'
    ans = split_into_chunks(text)
    print(ans)
    
    
if __name__ == '__main__':
    run()