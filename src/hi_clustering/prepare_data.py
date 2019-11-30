import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
from os import walk

globaldict = dict()

def print_dict(d):
    for key in list(d.keys()): 
        print(key, ":", d[key]) 

def read_data(textfile):
    d = dict()
    text = open(textfile).read().split()
    
    for word in text:
        word = word.lower()

        # make sure that word has only letters and numbers and _
        word = re.sub(r'\W+', '', word)
        
        # check if word contains only numbers
        word_wo_numbers = re.sub(r'[0-9]+', '', word)        
        if word_wo_numbers == "":
            continue 

        # Check if the word is already in dictionary 
        if word in d: 
            d[word] += 1
        else: 
            d[word] = 1
        
    # for key in list(d.keys()): 
    #     print(key, ":", d[key]) 
    
    return d

special_letters = ['a', 'e', 'i', 'o', 'u', 'y', 's']

def update_global_dict(d):
    # Checks if word is in global dictionary, if not, add.
    # Updates global dictionary values.
    for word in list(d.keys()):    
        val = d[word]
        if len(word) > 3:
            # some similar words detection
            wending = word[len(word) - 1]
            if wending == "s":
                if word[len(word) - 2] not in special_letters: 
                    word = word[0:len(word) - 1]
        
        if word in globaldict: 
            gval = globaldict[word]
            gval[0] += val
            gval[1] += 1
        else: 
            globaldict[word] = [val, 1]

def prepare_data(path, limit=0):
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        l = 0
        if limit == 0:
            l = len(filenames)
        else:
            l = min(limit, len(filenames))    
        f.extend(filenames[:l])
        break
    # update global dictionary with data from each file
    for text in f:
        d = read_data(path+'/'+text)
        update_global_dict(d)

# prepare_data("./sport", 100)

# print_dict(globaldict)


# read_data()


