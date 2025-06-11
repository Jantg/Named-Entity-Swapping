import os
import re

import numpy as np
import pandas as pd

import spacy
nlp = spacy.load("en_core_web_trf")
from toolz import curry
from functools import reduce
from pandarallel import pandarallel
pandarallel.initialize()
import scipy
import random
from itertools import combinations, product


@curry
def getEntlists(model, index, inv_map, ner_categories, excl_list, clst_idx):
    clst_txt = [(val, inv_map[re.search(r'(Company.*).pdf', index.docstore.docs.get(val).get_metadata_str()).group(1)],
    		 index.docstore.get_document(val).get_content()) \
                for idx,val in enumerate(index.docstore.docs.keys()) \
                if np.argmax(model.W,axis = 1)[idx] == clst_idx]
    entities_list = []
    excl_list = excl_list #['SG&A','FX','P&L','MG&A','CapEx','PLS','SEC',
                # 'Q&A','OpEx','DSO','FY','ForEx','FAS/CAS','OCI',
                #'EUR','OI','EBITDA']
    ner_categories = ner_categories #['PERSON','ORG','GPE','PRODUCT','LOC','EVENT']
    for idx, txt in enumerate(clst_txt):
        entities = []
        doc = nlp(txt[2])
        for ent in doc.ents:
            if bool(re.search(r'.*S&P Global.*',ent.text)):
                continue
            if ent.label_ in ner_categories:
                #if ent.label_ == 'PRODUCT':
                #    ent.label_ = 'ORG'
                if ent.label_ == 'LOC':
                    ent.label_ = 'GPE'
                entities.append((ent.text,ent.label_))
        ent_set = set(entities)
        # first match some words to exclude with patters
        ent_set = [e for e in ent_set if not bool(re.search(r'.*(EPS)|(Q\d)|(Slide\s*\d+)|(GAAP)|(\[Company Name\])|(\[Executive\d+\]).*',e[0]))]
        ent_set = [e for e in ent_set if not e[0] in excl_list]
        #ent_urls = [(str(e),'URL') for e in doc if e.like_url]
        #if len(ent_urls)!=0:
        #    ent_set = ent_set + ent_urls
        entities_list.append([txt[0],txt[1],clst_idx]+ent_set)
    return(entities_list)

def gradient(theta,alpha,freqs,n,u):
    dL_dtheta = 1/reduce(lambda x,y: x+y, theta+np.arange(1,u)*alpha, 0) - 1/reduce(lambda x,y: x+y, theta + np.arange(1,n))
    dL_dalpha = reduce(lambda x,y: x+y, np.arange(1,u)/(theta+np.arange(1,u)*alpha),0) -\
                reduce(lambda x,y: x+y, np.array([freqs[freqs.index == x].iloc[0]*np.sum(1/(np.arange(1,x)-alpha)) for x in freqs.index if x!=1]),0)
    return(np.array([dL_dtheta,dL_dalpha]))

def hessian(theta,alpha,freqs,n,u):
    d2L_dtheta2 = -1/reduce(lambda x,y: x+y, (theta+np.arange(1,u)*alpha)**2, 0) + 1/reduce(lambda x,y: x+y, (theta + np.arange(1,n))**2)
    d2L_dalpha2 = -reduce(lambda x,y: x+y, (np.arange(1,u)/(theta+np.arange(1,u)*alpha))**2,0) -\
                reduce(lambda x,y: x+y, np.array([freqs[freqs.index == x].iloc[0]*np.sum(1/(np.arange(1,x)-alpha)**2) for x in freqs.index if x!=1]),0)
    d2L_dtheta_dalpha = -reduce(lambda x,y: x+y, (np.arange(1,u)/(theta+np.arange(1,u)*alpha)**2),0)
    return(np.array([[d2L_dtheta2,d2L_dtheta_dalpha],[d2L_dtheta_dalpha, d2L_dalpha2]]))
    
def newton_optimize(theta_init, alpha_init ,freqs,n,u, tol, max_iter):
    theta, alpha = theta_init, alpha_init
    for idx in range(max_iter):
        grad = gradient(theta, alpha,freqs,n,u)
        hess = hessian(theta, alpha,freqs,n,u)
        if np.linalg.norm(grad) < tol:  # Convergence check
            break
        delta = np.linalg.solve(hess, -grad)  # Newton update step
        theta, alpha = theta + delta[0], alpha + delta[1]
        #print([theta,alpha])
        #if idx%100 == 0:
            
            #print(idx)
    return(theta, alpha)

def fitEwensPitman(df, tol = 1e-7, max_iter = 100):
	n = df.apply(lambda x: set(x),axis = 1).value_counts().sum()
	u = df.apply(lambda x: set(x),axis = 1).value_counts().shape[0]
	freqs = df.apply(lambda x: set(x),axis = 1).value_counts().value_counts().sort_index()
	try:
	    s1 = freqs[freqs.index == 1].iloc[0]
	except IndexError:
	    s1 = 0
	try:
	    s2 = freqs[freqs.index == 2].iloc[0]
	except IndexError:
	    s2 = 0
	c = (s1*(s1-1))/s2
	theta_init = (n*u*c-s1*(n-1)*(2*u+c))/(2*s1*u+s1*c-n*c)
	alpha_init = (theta_init*(s1-n)+(n-1)*s1)/(n*u)
	theta_opt, alpha_opt = newton_optimize(theta_init,alpha_init,freqs,n,u,tol,max_iter=100)
	return theta_opt, alpha_opt
	
def if_trueswap(idxs,df_group,swapvars):
    idx1,idx2 = idxs
    if any([df_group.iloc[idx1][attr] =='' for attr in swapvars]) or any([df_group.iloc[idx2][attr] =='' for attr in swapvars]):
        return(False)
    # if different filename_swap  
    ticker1 = re.search(r'([A-Z]+).*\.pdf',df_group.iloc[idx1]['File Name']).group(1)
    ticker2 = re.search(r'([A-Z]+).*\.pdf',df_group.iloc[idx2]['File Name']).group(1)
    if ticker1 == ticker2:
        return(False)
    else:
        if df_group.iloc[idx1].drop(['Clst_idx','Key','File Name']+swapvars).\
        equals(df_group.iloc[idx2].drop(['Clst_idx','Key','File Name']+swapvars)):
            return(False)
        else:
            if any([df_group.iloc[idx1][attr] == df_group.iloc[idx2][attr] for attr in swapvars]):
                return(False)
            else:
                return(True)


def swapOnce(df, swapvars):
    true_swaps = df.groupby('Clst_idx')\
                   .parallel_apply(lambda x: [idx for idx,val in enumerate([if_trueswap(idxs,x,list(swapvars))\
                                               for idxs in list(combinations(range(len(x)),2))]) if val])
    true_swap_distri = true_swaps.apply(lambda x: len(x))
    if true_swap_distri.sum() == 0:
        return None
    swap_idx = random.sample(range(true_swap_distri.sum()),1)[0]
    s = np.cumsum(true_swap_distri[true_swap_distri!=0])>swap_idx
    clst_idx = s[s].index[0]
    pair_idx = true_swaps[true_swaps.index == clst_idx].iloc[0][-int(np.cumsum(true_swap_distri)[np.cumsum(true_swap_distri).index == clst_idx]-swap_idx)]
    idx1, idx2 = list(combinations(range(df[df['Clst_idx'] == clst_idx].shape[0]),2))[pair_idx]
    swapped = df[df['Clst_idx'] == clst_idx].iloc[[idx1,idx2]]             
    return df.drop(swapped.index), swapped.index

def riskMeasure(risk_measure,numsingleton, N, n, s1, alpha, theta):
    pprop =(n/s1*\
    np.exp((alpha-1)*np.log(N)+scipy.special.loggamma(theta+1)-scipy.special.loggamma(theta+alpha)))
    risk_measure = risk_measure-(numsingleton/s1*pprop)
    return risk_measure
 
