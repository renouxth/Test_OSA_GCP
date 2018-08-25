# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:39:32 2018

@author: Marie G
"""

import pandas as pd

import numpy as np
import datetime

from datetime import date, timedelta
from openpyxl import load_workbook

from Ventes import bdd_produits
from Ventes import data_file
#from communicationstock import writer

d=(date.today() - timedelta(days=500))
data_inno=data_file[data_file['Date']>=(d)]
data_inno["Date de lancement"] = data_inno['EAN 13'].map(bdd_produits.set_index('EAN 13')['Date de lancement'])
data_inno=data_inno.loc['1900-01-01':'3999-01-01']
data_inno=data_inno[data_inno['Date de lancement']>= datetime.datetime(d.year, d.month, d.day)]
data_inno=pd.pivot_table(data_inno, values='Ventes Unites', index=['Magasin', 'Produit', 'EAN 13'], aggfunc=np.sum)


data_inno=data_inno.reset_index()
data_inno['Magasin']=data_inno['Magasin'].str.replace('/', '.')
print(data_inno[data_inno['Magasin']=='CARREFOUR BEAUNE'])

data_inno=data_inno.sort_values(by=['Ventes Unites'], ascending=False)
data_inno=data_inno.sort_values(by=['Magasin'])


data_inno=data_inno.set_index('Magasin')



"""

magasin=""
for x in data_inno.index :

    if magasin == x:
        continue
    else :
        print(data_inno[data_inno.index==x])
        #writer = pd.ExcelWriter('SCORECARD/stock_scorecard_'+ x +'.xlsx', engine='xlsxwriter')
        #data_i[data_i.index==x].reset_index()[['Produit', 'EAN 13', 'Stock faux']].iloc[:10].to_excel(writer, sheet_name='TOP Stock faux')
        #writer.save()
        book=load_workbook('SCORECARD/stock_scorecard_'+ x +'.xlsx')
        writer = pd.ExcelWriter('SCORECARD/stock_scorecard_'+ x +'.xlsx', engine='openpyxl')
        writer.book = book
        data_inno[data_inno.index==x].reset_index()[['Produit', 'EAN 13', 'Ventes Unites']].iloc[:10].to_excel(writer, sheet_name="TOP Innovation", index=False)
        writer.save()
        writer.close()
        magasin=x

        
"""