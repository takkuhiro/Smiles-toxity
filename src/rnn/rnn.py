from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

import torch
import torch.nn as nn
import random
import time
import math
import pickle
import numpy as np

def findFiles(path): return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c)!='Mn' and c in all_letters)

def readLines(max_length, filename):
    #lines = open(filename, encoding='utf-8').read().strip().split('\n')
    #return [unicodeToAscii(line) for line in lines]
    #lines = open(filename, 'r').read().strip().split('\n')
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')
    tmp = []
    for i, line in enumerate(lines):
        #if i % 100 == 0:print(i)
        if len(line) > max_length:
            tmp.append(line[:max_length])
        else:
            tmp.append(line)
      
    return tmp

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden
        
    def initHidden(self):
        return torch.zeros(1, self.hidden_size) 

def randomChoice(I):
    return I[random.randint(0, len(I) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor
                 
# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor
                                                  
# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze(-1)
    hidden = rnn.initHidden()
    rnn.zero_grad()
    loss = 0
    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        tmp = target_line_tensor[i].unsqueeze(0)
        l = criterion(output, tmp)
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def sample(rnn, category, start_letter='C'):
    with torch.no_grad():
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter
        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)
        return output_name

def samples(rnn, category, start_letters='CCC'):
    text = ''
    for start_letter in start_letters:
        text += sample(rnn, category, start_letter) + '\n'
        #print(text)
    return text

if __name__=='__main__':
    files = ['../w0.txt', '../w1.txt']
    mode = 'train_test' #train, test, train_testのいずれかを設定
    outputfile0 = 'out0-2.txt'
    outputfile1 = 'out1-2.txt'
    modelfile_name = 'model2.pickle'
    category_lines = {}
    all_categories = [0, 1]
    #all_letters = string.ascii_letters + " .,;'-"
    max_length = 128
    output_num = 10000
    all_letters = ''
    first_char_box = []
    with open('../word2id.pickle', 'rb') as f:
        w2i = pickle.load(f)
    #print(w2i)
    for k in w2i.keys():
        all_letters += k
    n_letters = len(all_letters) + 1 # Plus EOS marker
    n_categories = len(all_categories)

    if mode in ['train', 'train_test']:
        criterion = nn.NLLLoss()
        learning_rate = 0.0005
        lossfile = 'loss2.pickle'
        for (filename, category) in zip(files, all_categories):
            lines = readLines(max_length, filename)
            category_lines[category] = lines
            for line in lines:
                if line[0] not in first_char_box:
                    first_char_box.append(line[0])
        rnn = RNN(n_letters, 128, n_letters)

        n_iters = 10000
        print_every = 50
        plot_every = 50
        all_losses = []
        total_loss = 0
        start = time.time()

        for iter in range(1, n_iters+1):
            output, loss = train(*randomTrainingExample())
            total_loss += loss
            if np.isnan(total_loss):
                print("loss      : ", loss)
                print("total_loss: ", total_loss)
                print("output    : ", output)
            if iter % print_every == 0:
                print('%s (%d %d%%)%.4f' %(timeSince(start), iter, iter / n_iters * 100, loss))
            if iter % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0

        #import matplotlib.pyplot as plt
        #import matplotlib.ticker as ticker
        #plt.figure()
        #plt.plot(all_losses)
        
        with open(modelfile_name, mode='wb') as f:
            pickle.dump(rnn, f)
        #後でグラフ出力するために保存
        with open(lossfile, 'wb') as f:
            pickle.dump(all_losses, f)
    
    if mode in ['train_test', 'test']:
        for (filename, category) in zip(files, all_categories):
            lines = readLines(max_length, filename)
            category_lines[category] = lines
            for line in lines:
                if line[0] not in first_char_box:
                    first_char_box.append(line[0])
        rnn = RNN(n_letters, 128, n_letters)
        with open(modelfile_name, mode='rb') as f:
            rnn = pickle.load(f)
        
        with open(outputfile0, 'w') as f:
            choices = ''
            for _ in range(output_num):
                choices += random.choice(first_char_box) 
            f.write(samples(rnn, 0, choices))
            f.flush()

        with open(outputfile1, 'w') as f:
            choices = ''
            for _ in range(output_num):
                choices += random.choice(first_char_box)
            f.write(samples(rnn, 1, choices))
            f.flush()

