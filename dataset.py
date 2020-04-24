"""
Created on Fri Apr 24 18:31:55 2020

@author: Patrik Tariška
"""
# import potrebných knižníc
import pandas as pd
import numpy as np

# import datasetu pomocou pandas
stars_data_raw = pd.read_csv("stars_data.csv")

# print zakladných info datasetu
print("Počet riadkov: ", stars_data_raw.shape[0])
print("Počet stĺpcov: ", stars_data_raw.shape[1])
print("Názvy stĺpcov: ", stars_data_raw.columns)
print("Prvých 10 riadkov datasetu: \n")
print(stars_data_raw.head(10))
print("Posledných 10 riadkov datasetu: \n")
print(stars_data_raw.tail(10))
