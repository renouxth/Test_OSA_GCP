# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 18:28:01 2018

@author: Marie G
"""

import pandas as pd

import numpy as np

import Ventes

from datetime import date, timedelta


from Ventes import data_file



data_stock=data_file[data_file['Date']>=(date.today() - timedelta(days=500))]
data_stock=pd.pivot_table(data_stock, values='Stock', index=['Magasin', 'Produit', 'EAN 13'], columns='Date', aggfunc=np.sum)
data_stock['Stock min']=data_stock.min(axis=1)
data_stock=data_stock[data_stock['Stock min']!=0]



    
#data_stock=data_stock.ix[data_stock[].count(data_stock)>=3]
data_i=pd.DataFrame(columns=['Magasin', 'Produit', 'EAN 13', 'Stock faux'])
n=0
for indexx, row in data_stock.iterrows() :
    compteur=0
    
    for i in row:
        
        if i==row['Stock min']:
            print(indexx)
            print(i)
            compteur = compteur+1
            
            if compteur>=3:
                
                data_i=data_i.append(pd.Series(indexx, index=['Magasin', 'Produit', 'EAN 13']), ignore_index=True)
               
                data_i['Stock faux'].values[n]=row['Stock min']

                n=n+1
                break
    else:
        continue
    

data_i['Magasin']=data_i['Magasin'].str.replace('/', '.')
data_i=data_i.sort_values(by=['Stock faux'], ascending=False)
data_i=data_i.set_index('Magasin')
print(data_i)


magasin=""
for x in data_i.index :
    if magasin == x:
        continue
    else :
        print(data_i[data_i.index==x])
        writer = pd.ExcelWriter('SCORECARD/stock_scorecard_'+ x +'.xlsx', engine='xlsxwriter')
        data_i[data_i.index==x].reset_index()[['Produit', 'EAN 13', 'Stock faux']].iloc[:10].to_excel(writer, sheet_name='TOP Stock faux', index=False)
        writer.save()
        magasin=x
    
