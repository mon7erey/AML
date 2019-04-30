#!/usr/bin/env python
# coding: utf-8

# In[1]:


### multi-processing
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


# In[2]:
### multi-processing function
def imap_unordered_bar(func, args, n_processes = 2): 
    p = Pool(n_processes)
    res_list = []
    with tqdm(total = 1, dynamic_ncols=True) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args)), disable=True):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list

### make a nested list for computing, e.g., [[],[],[]] -> a list with three sub list
def comparison_list(len_of_ref_seq):
    tmp_list = []
    for i in range(0,len_of_ref_seq):
        tmp_list.append([])
    return tmp_list


### for each account in the alert/general list, calculate the similarity with other accounts (an account is formed as time series)
def cluster_matching(vec_be_calculated, vec_as_ref, be_i, as_j, accept_range, accept_ratio):
    len_of_vec = len(vec_be_calculated)
    result_list = []
    match = 0
    for i in range(len_of_vec):
        seq_diff = abs(ord(vec_be_calculated[i]) - ord(vec_as_ref[i]))
        if(seq_diff <= accept_range): # e.g., 'a' in ASCII = 97; 'b' in ASCII = 98; abs(97 - 98) = 1
            match += 1
        else: 
            match -= round(4/(1+np.exp(-0.8*seq_diff)),2)
            
    similarity = round((2*match)/(2*len_of_vec),2)
    if (similarity > accept_ratio):
        result_list.append([be_i, as_j,similarity]) ### symmetric match
        result_list.append([as_j, be_i, similarity]) ### symmetric match
        
    return result_list

### make a matrix for parallel computing
def build_triangular_mx(len_of_ref_seq):
    parallel_ck = np.zeros(shape=(int((len_of_ref_seq*(len_of_ref_seq-1))/2) ,2)) ### for parallel function distributes tasks
    k = 0
    for i in range(0,len_of_ref_seq):
        for j in range(i,len_of_ref_seq):
            if i != j:
                parallel_ck[k][0] = int(i) ### 存依序計算的波形i
                parallel_ck[k][1] = int(j) ### 存依序計算的波形j
                k= k+1
    return parallel_ck

### a target account that we'd like to know if it has some other similar ts pattern w.r.t an account
### e.g., target_ts id: 8 and its similar id with similarity score: [3, 0.93], [16, 0.9], [5, 0.87], [15, 0.83]
### [3, 0.93], [16, 0.9], [5, 0.87], [15, 0.83] is formed as query_ts
def seq_matching(result_mapping, vec_be_calculated, accept_range, accept_ratio,target_ts, ref_seq):
    len_of_vec = len(vec_be_calculated)
    tmp_list = []
    for j in range(len(ref_seq)):
        match = 0
        for i in range(len_of_vec):
            seq_diff = abs(ord(vec_be_calculated[i]) - ord(ref_seq[j][i]))
            if(seq_diff <= accept_range): # e.g., 'a' in ASCII = 97; 'b' in ASCII = 98; abs(97 - 98) = 1
                match += 1
            else: 
                match -= round(4/(1+np.exp(-0.8*seq_diff)),2)
        similarity = round((2*match)/(2*len_of_vec),2)        
        if (similarity > accept_ratio and j != target_ts):
            tmp_list.append([j, similarity]) # [id , matching ratio]
    return tmp_list

### sort by first element (column)
def takeFirst(elem):
    return elem[0]

### sort by second element (column)
def takeSecond(elem):
    return elem[1]

### get all the target_ts's query_ts and then get query_ts's similar ts id
### make a set of all the query_ts's similar ts id tworads the target_ts
### e.g., target_ts id: 8 -> the set of all the query_ts's similar ts id
def id_group_extraction(result_mapping, query_ts, target_ts):
    id_extraction = []
    for i in query_ts:
        tmp_id_extraction = [row[0] for row in result_mapping[i[0]]]
        id_extraction += tmp_id_extraction
    suspicious_cluster = set(id_extraction)
    if len(set(id_extraction))>=1:
        suspicious_cluster.remove(target_ts)
    return sorted(suspicious_cluster)

### build a report for a target_ts list. target_ts is matched to an alert/general list
### the alert/general list in which an account is with its own similar account pattern. (i.e.,the list runs cluster_matching function)
def scan_suspicious_clusters(result_mapping, target_ts, ref_seq, accept_fuzzy_range, accept_fuzzy_ratio):
    vec_be_calculated = ref_seq[target_ts]
    query_ts = seq_matching(result_mapping, vec_be_calculated, accept_fuzzy_range, accept_fuzzy_ratio, target_ts, ref_seq)
    query_ts.sort(key=takeSecond, reverse=True) ### most similar pattern with the target_ts
    suspicious_cluster = id_group_extraction(result_mapping, query_ts, target_ts)
    return [target_ts, query_ts ,suspicious_cluster]