#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:00:11 2016

@author: student
"""

def extractUnique(data):
    ans = set()
    for i in data:
        t = i.split('/')
        for j in t:
            if j != '':
                ans.add(j)
    ans = list(ans)
    ans = map(int,ans)
    return ans

def dummies(data,array):
    for i in range(len(data)):
        t = data[i].split('/')
        for j in t:
            if j != '':
                array[i][int(j)] = 1
    return array
    
import pandas as pd
import numpy as np

user_data = pd.read_csv('user_info.txt',delim_whitespace=True,header=None)
ques_data = pd.read_csv('question_info.txt',delim_whitespace=True,header=None)

expert_tags = user_data[1]
unique_tags = extractUnique(expert_tags)

word_tags =user_data[2]
unique_words = extractUnique(word_tags)
unique_words.sort()

char_tags =user_data[3]
unique_chars = extractUnique(char_tags)
unique_chars.sort()

ques_word_tags = ques_data[2]
unique_ques_words = extractUnique(ques_word_tags)

ques_char_tags = ques_data[3]
unique_ques_chars = extractUnique(ques_char_tags)
unique_ques_chars.sort()

u_chars = np.zeros((user_data.shape[0],max(unique_chars)+1),int)
c = dummies(char_tags,u_chars)
dummy_chars = pd.DataFrame(c)

dummy_chars = dummy_chars.loc[:, (dummy_chars != 0).any(axis=0)]                         
                           
u_tags = np.zeros((user_data.shape[0],max(unique_tags)+1),int)
z = dummies(expert_tags,u_tags)
dummy_tags = pd.DataFrame(z)
dummy_tags = dummy_tags.loc[:, (dummy_tags != 0).any(axis=0)]
                            
u_words = np.zeros((user_data.shape[0],max(unique_words)+1),int)
w = dummies(word_tags,u_words)
dummy_words = pd.DataFrame(w)
dummy_words = dummy_words.loc[:, (dummy_words != 0).any(axis=0)]
                            
new_user_data = user_data.drop([1,2,3],axis=1)

new_user_data = pd.concat([new_user_data,dummy_tags],axis=1)
new_user_data = pd.concat([new_user_data,dummy_chars],axis=1)
new_user_data = pd.concat([new_user_data,dummy_words],axis=1)
new_user_data.to_csv('user_data.csv',index=False)

new_ques_data = ques_data.drop([2,3],axis=1)

u_ques_chars = np.zeros((ques_data.shape[0],max(unique_ques_chars)+1),int)
a = dummies(ques_char_tags,u_ques_chars)
dummy_ques_chars = pd.DataFrame(a)
dummy_ques_chars = dummy_ques_chars.loc[:, (dummy_ques_chars != 0).any(axis=0)]
                                        
u_ques_words = np.zeros((ques_data.shape[0],max(unique_ques_words)+1),int) 
q = dummies(ques_word_tags,u_ques_words)
dummy_ques_words = pd.DataFrame(q)
dummy_ques_words = dummy_ques_words.loc[:, (dummy_ques_words != 0).any(axis=0)]                                      
                                        
new_ques_data = pd.concat([new_ques_data,dummy_ques_chars],axis=1)
new_ques_data = pd.concat([new_ques_data,dummy_ques_words],axis=1)
new_ques_data.to_csv('ques_data.csv',index=False)