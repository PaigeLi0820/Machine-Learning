# -*- coding: utf-8 -*-
"""
Name: Li, Pei-En
NetID: lipeien2

Created on Sun Apr  5 21:26:03 2020
@author: paige
"""
import nltk
from collections import defaultdict
from stop_list import closed_class_stop_words
import math
import string
import copy
from numpy import linalg as LA
from sklearn.metrics.pairwise import cosine_similarity

def ReadInput(filename):
	f = open(filename, "r")
	read_flag = 0;
	start_flag = 0;
	tmp_str = ""
	tokens = []
	for line in f:
		if line.startswith(".I"):
			read_flag = 0
			if start_flag == 0:
				start_flag = 1
			else:
				tokens.append(tmp_str)
				tmp_str = ""
		elif line.startswith(".W"):
			read_flag = 1
		elif read_flag == 1:
			tmp_str = tmp_str + line
	if len(tmp_str) > 0:
		tokens.append(tmp_str)
	f.close()
	return tokens

def SplitTokens(tokens_str):
	tokens_ret = []
	stop_punc = list(string.punctuation)
	stop_list = closed_class_stop_words + stop_punc
	for token in tokens_str:
		token_split = nltk.word_tokenize(token)
		tmp = []
		for word in token_split:
			if word.lower() not in stop_list:
				if not word.isdigit():
					tmp.append(word)
		tokens_ret.append(tmp)
	return tokens_ret

'''	
#This is fine, but it takes longer time to run.
def Calc_IDF(tokens):
	flatten_tokens = [y for x in tokens for y in x]
	word_set = set(flatten_tokens)
	tmp_dic = defaultdict(int)
	ret_dic = defaultdict(int)
	total_docs = len(tokens)
	for word in word_set:
		for token in tokens:
			if word in token:
				tmp_dic[word] += 1
	for word in tmp_dic:
		ret_dic[word] = math.log(float(total_docs)/float(tmp_dic[word]))
	return ret_dic
'''
#This is faster
def Calc_IDF(docs):
	dic = {}
	ret_dic = {}
	doc_idx = 0
	docs_num = len(docs)
	for doc in docs:
		for word in doc:
			if word not in dic:
				dic[word] = [0]*docs_num
				dic[word][doc_idx] = 1
			else:
				dic[word][doc_idx] += 1
		doc_idx += 1
	for word in dic:
		for i in range(docs_num):
			if dic[word][i] != 0:
				if word in ret_dic:
					ret_dic[word] += 1
				else: 
					ret_dic[word] = 1
	for word in ret_dic:
		ret_dic[word] = math.log(docs_num/float(ret_dic[word]))
	return ret_dic

def Calc_TF(docs):
	ret_list = []
	tmp_dic = {}
	for doc in docs:
		tmp_dic.clear()
		for word in doc:
			if word in tmp_dic:
				tmp_dic[word] += 1
			else:
				tmp_dic[word] = 1
		ret_list.append(copy.copy(tmp_dic))
	return ret_list

def Calc_TF_IDF(tokens, IDF, TF):
	ret_vec = []
	tmp = float(0)
	for token in tokens:
		if token in TF:
			if TF[token] > 0:
				tmp = IDF[token] * (1 + math.log(TF[token]))
			else:
				tmp = 0
		else:
			tmp = 0 
		ret_vec.append(tmp)
	return ret_vec

def Calc_Cossim(vec1, vec2):
	vec1_len = LA.norm(vec1)
	vec2_len = LA.norm(vec2)
	dot = 0
	for i in range(len(vec1)):
		dot += vec1[i] * vec2[i]
	denominator = vec1_len*vec2_len
	if denominator != 0:
		return float(dot) / float(vec1_len*vec2_len)
	else:
		return 0
'''
def Calc_Cossim(vec1, vec2):
    cosAB = cosine_similarity(vec1, vec2, dense_output=True)
    return cosAB 
'''	
def WriteAns(filename, results):
	f = open(filename, "w")
	for result in results:
		tmp_str = str(result[0]) + " " + str(result[1]) + " " + str(result[2]) + "\n"
		f.write(tmp_str)
	f.close()	

if __name__ == "__main__":
    print("Read Query")
    qStr = ReadInput("cran.qry")
    print("Read Abstract")
    aStr = ReadInput("cran.all.1400")
	
    print("Split Query")
    qGood = SplitTokens(qStr)
    print("Split Abstract")
    aGood = SplitTokens(aStr)
	
    print("Calculate Query IDF")
    qIDF = Calc_IDF(qGood)
    print("Calculate Abstract IDF")
    aIDF = Calc_IDF(aGood)
	
    print("Calculate Query TF")
    qTF = Calc_TF(qGood)
    print("Calculate Abstract TF")
    aTF = Calc_TF(aGood)
	
    print("Calculate query vector of TF_IDF")
    qVec = []
    idx = 0
    for query in qGood:
        qVec.append(Calc_TF_IDF(query, qIDF, qTF[idx]))
        idx += 1
	
    print("Compute Cosine Similarity")
    abs_num = len(aGood)
    qidx = 0
    tup_list = []
    result_list = []
    for query in qGood:
        tmp_vec = qVec[qidx]
        for aidx in range(abs_num):
            aVec = Calc_TF_IDF(query, aIDF, aTF[aidx])
            cos_sim = Calc_Cossim(tmp_vec, aVec)
            tmp_tup = (qidx+1, aidx+1, cos_sim)
            tup_list.append(tmp_tup)
        qidx += 1
        tup_list.sort(key = lambda tup: tup[2], reverse = True)
        result_list = result_list + tup_list
        tup_list = []
	#Write result_list to output file
    print("Write Answer to output.txt...")
    WriteAns("output.txt", result_list)
    print("All Done!")
