# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 17:29:23 2021

@author: ASUS PC
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

raw_data = pd.read_excel("raw_data.xlsx")
raw_data = raw_data.iloc[:,5:]
correlation = raw_data.corr(method='pearson')
f, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(correlation, center = 0,  linewidths=.1)
#t = sns.pairplot(correlation)