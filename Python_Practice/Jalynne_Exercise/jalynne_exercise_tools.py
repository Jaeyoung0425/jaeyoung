#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:12:24 2021

@author: twodigit
"""

# %%
import re
# %%


def arrow_clear(list_):
    result_list = []
    for temp_result in list_:
        if bool(re.match('.*→.*', temp_result)) == True:
            point = [i.start() for i in re.finditer('→', temp_result)]
            result_list.append(temp_result[point[-1] + 1:].strip())
        else:
            result_list.append(temp_result)
    return result_list
# %%


def parenthesis_clear(list_):
    result_list = []
    for temp_elem in list_:
        try:
            r = temp_elem.replace("[", "(").replace("]", ")").replace(
                "{​​​​​​​", "(").replace("}​​​​​​​", ")").replace(" ", "")
            r = re.sub(r'\([^)]*\)', '', r).replace(')', '').replace('(', '')
            result_list.append(r)
        except:
            result_list.append('no result')

    return result_list
# %%


def synonym_clear(list_, synonym_dic):
    result_list = []
    for r in list_:
        flag = 0
        for key in list(synonym_dic.keys()):
            synonym_list = synonym_dic[key]
            for synonym in synonym_list:
                if synonym in r:
                    result_list.append(key)
                    flag = 1
                    break
            if flag == 1:
                break
        if flag == 1:
            pass
        else:
            result_list.append('no result')
    return result_list

# %%

def specialchar_clear(list_):
    result_list = []
    for s in list_:
        try:
            s = re.sub(r'[^ ㄱ-ㅣ가-힣A-Za-z]', '', s)
            result_list.append(s)
        except:
            result_list.append(s)
    return result_list
# %%


def hangul_clear(list_):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result_list = []
    for temp_elem in list_:
        try:
            h = hangul.sub('', temp_elem)
            result_list.append(h)
        except:
            result_list.append(temp_elem)
    return result_list
# %%

def eng_clear(list_):
    eng = re.compile('[^ a-z A-Z]+')
    result_list = []
    for temp_elem in list_:
        try:
            h = eng.sub('', temp_elem)
            result_list.append(h)
        except:
            result_list.append(temp_elem)
    return result_list

#%%

def date_8digits(list_):
    result_list = []
    for temp_elem in list_:
        try:
            temp_elem = temp_elem.replace(' ','')
            point = temp_elem.find('년')
            point1 = temp_elem.find('월')
            point2 = temp_elem.find('일')
            try:
                if len(temp_elem[point + 1:point1]) == 1:
                    temp_elem = temp_elem[:point + 1] + '0' + temp_elem[point1 - 1:]
                    
                    if len(temp_elem[point1 + 2:point2 + 1]) == 1:  
                        temp_elem = temp_elem[:point1 + 2] + '0' + temp_elem[point2:]
                        result_list.append(temp_elem)

                    else:  
                        result_list.append(temp_elem)

                else:  
                    temp_elem = temp_elem[:point1 + 1] + '0' + temp_elem[point2 - 1:]
                    result_list.append(temp_elem)
            except:
                result_list.append(temp_elem)
        except:
            result_list.append(temp_elem)
            
    return result_list
        
#%%

def date_clear(list_):
    result_list=[]
    for temp_elem in list_:
        try:
            temp_elem = re.sub(r'\([^)]*\)', '', temp_elem)
            temp_elem = temp_elem.replace("개최되지않음", "").replace("년", "").replace("월", "").replace("일", "")
            result_list.append(temp_elem)
        except:
            result_list.append(temp_elem)
           
    return result_list


