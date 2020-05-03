"""
Created on Fri Apr 24 18:31:55 2020

@author: Patrik Tariška
"""
# import potrebných knižníc
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.utils import to_categorical

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

train_x = stars_data_raw.iloc[:,1:4].values
train_y = stars_data_raw.iloc[:, 4].values

categorical = np_utils.to_categorical(train_y)

model = Sequential()


model.add(Dense(50, input_dim=3, activation='sigmoid')) #vstupna vrstva
model.add(Dense(50, activation='sigmoid')) # skryta vrstva
model.add(Dense(50, activation='sigmoid')) # skryta vrstva
model.add(Dense(6, activation = 'sigmoid')) # vystupne neurony

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
hodnoty=[]
# zbehnutie siete 20 krat
# ulozime vysledne skora
for _ in range(20):
    model.fit(train_x, categorical)
    scores = model.evaluate(train_x, categorical)
    hodnoty.append(scores[1]*100)

# vypisanie maximalneho a minimalneho skora uspesnosti siete
print("Maximálne skóre: {:.2f}%".format(max(hodnoty)))
print("Minimálne skóre: {:.2f}%".format(min(hodnoty)))
# sigmoid neoptimalna funkcia pre tuto siet
# maximalna uspesnost siete 16.67% - neuspokojive
# optimizer sgd

#%% 3.5. 2020 Zvysenie poctu iteracii ucenia (atribut epochs v metode fit)
# znova otestujeme na 20 pokusoch
hodnoty = []

for _ in range(20):
    # pocet epoch nastavime na 5
    model.fit(train_x, categorical, epochs=5)
    scores = model.evaluate(train_x, categorical)
    hodnoty.append(scores[1]*100)

# vypisanie maximalneho a minimalneho skora uspesnosti siete
print("Maximálne skóre: {:.2f}%".format(max(hodnoty)))
print("Minimálne skóre: {:.2f}%".format(min(hodnoty)))

# po zvyseni poctu epoch na 5 sa nam podarilo v niektorych pripadoch zvysit 
# uspesnost siete na maximalne 33.33%, z toho nam vyplyva ze pocet trenovani ma efekt 
# na vyslednu uspesnost siete, avsak stale su vysledne hodnoty neuspokojive
# pricinami moze byt nevhodne rozdelenie siete (pocet neuronov vo vrstvach), 
# nevhodna aktivacna funkcia, nevhodne data (nie je mozna efektivna
# klasifikacia), alebo nevhodny optimizer

#%% 3.5. 2020 Ponechanie predchadzajuceho procesu trenovania so zmenenou strukturou
# siete (3 vstupy, 3 skryte vrstvy po 100 neuronoch a 6 vystupov)
# aktivacnu funkciu nateraz nechavame rovnaku teda sigmoid, rovnako aj atributy 
# metody compile

model2 = Sequential()
model2.add(Dense(100, input_dim=3, activation='sigmoid')) #vstupna vrstva
model2.add(Dense(100, activation='sigmoid')) # skryta vrstva
model2.add(Dense(100, activation='sigmoid')) # skryta vrstva
model2.add(Dense(6, activation = 'sigmoid')) # vystupne neurony
model2.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
hodnoty=[]

for _ in range(20):
    # pocet epoch nastavime na 5
    model.fit(train_x, categorical, epochs=5)
    scores = model.evaluate(train_x, categorical)
    hodnoty.append(scores[1]*100)

# vypisanie maximalneho a minimalneho skora uspesnosti siete
print("Maximálne skóre: {:.2f}%".format(max(hodnoty)))
print("Minimálne skóre: {:.2f}%".format(min(hodnoty)))

# maximalna uspesnost siete bola 33.33% a teda nezmenena
# zvysenie poctu neuronov v skrytych vrstvach nam vsak pomohlo s konzistentnostou
# nasej siete, pretoze sme pravidelnejsie az takmer vzdy dosahovali maximalnu hodnotu
# uspesnosti
# dalej by bolo dobre sa sustredit na vhodnejsie aktivacne funkcie pripadne optimizer






