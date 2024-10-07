import numpy as np
# import tensorflow as tf
import torch
from torch.utils.data import Dataset
import os

MAX_TIMESTEPS = 100
path = "/home/volodimir/Bureau/ForLang/data/Large/"
name = "04.02.TLTT.2.3.0_Dev.txt"

def data_proces(path,name):
    data = []
    labels = []
    with open(path+name) as file:
        length = len(file.readlines())
    with open(path+name) as file:
        for j in range(length):
            line = file.readline()
            labeled_line = line.split()
            data.append(labeled_line[0])
            labels.append(labeled_line[1])
    return (data,labels)

def get_max_length(data):
    return np.max([len(x) for x in data])
    

def splitter(word):
    res = []
    for j in word:
        res.append(j)
    return res


def alphabet_extractor(data):
    alphabet = []
    for elem in data:
        for lettre in elem:
            alphabet.append(lettre)
    alphabet = set(alphabet)
    return sorted(list(alphabet))
    

def integer_embeder(alphabet,lettre):
    return alphabet.index(lettre)


def label_encoder(label):
    res = (label == 'TRUE')*1 + (label == 'FALSE')*0
    return res


def lettre_one_hot_encoder(one_hot_list,alphabet,lettre):
    indx = integer_embeder(alphabet,lettre)
    return one_hot_list[indx]



# ### test zone
# data, labels = data_proces(path,name)
# alphabet = alphabet_extractor(data)
# print(alphabet)
# word = data[0]
# print(word)
# for j in word:
#     integ = integer_embeder(alphabet,j)
#     print(integ)

# ### end test zone


def preprocess(line, nb_class): #In the preprocess, the x is the sentence, and the y is the sentence shifted one element left
    splitted_line = splitter(line)
    return lettre_one_hot_encoder(nb_class, splitted_line), splitted_line[1:]


def get_vocabulary_size(data): #get the vocabulary size, will be useful for the embedding layer
    alphabet = alphabet_extractor(data)
    return len(alphabet)


def load_spice_dataset(sample_name:str, path="data/Small/"): #Load the spice sample with the number given as args contained in the folder $path
    data, labels = data_proces(path,name)
    nb_class = get_vocabulary_size(data)
    lines = list(map(lambda x : preprocess(x, nb_class), lines)) #we preprocess every lines
    return lines

class Dataset(torch.utils.data.Dataset):#type: ignore
    def __init__(self,path,name,length=None):
        self.X, self.y = data_proces(path,name)
        self.alphabet = alphabet_extractor(self.X)
        self.target_alphabet = ['TRUE','FALSE']
        self.one_hot = []
        for j in range(len(self.alphabet)):
            self.one_hot.append(np.eye(len(self.alphabet))[:,j])
        
        if length == None:
            self.padding_length = get_max_length(self.X)
        else:
            self.padding_length = length
    def print_one_hot_alphabet(self):
        print(self.one_hot)
    
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.to_one_hot(self.X[index]), self.label_to_one_hot(self.y[index])#, dtype=torch.float32)
    def to_one_hot(self, x):
        x_one_hot = []
        for letter in x:
            x_one_hot.append(lettre_one_hot_encoder(self.one_hot,self.alphabet,letter))
        for _ in range(self.padding_length - len(x)):
            x_one_hot.append(np.zeros((len(self.alphabet),)))
        x_one_hot.append(len(x)*np.ones((len(self.alphabet),)))
        return torch.tensor(x_one_hot, dtype=torch.float32)

    def label_to_one_hot(self,y):
        y_one_hot = label_encoder(y)
        return torch.tensor(y_one_hot, dtype=torch.int16)
    def alphabet_len(self):
        return len(self.alphabet)