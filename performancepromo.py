# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:18:20 2018

@author: Marie G
"""
import pandas as pd
import numpy as np
import Ventes
from Ventes import newdatabase
from Ventes import bdd_produits
from Ventes import data_file

recap_cas=pd.read_excel(pd.ExcelFile(r"C:\Python27\Scripts\844182-Source-Code-10-25-2016\Source Code\Nouveau projet\Recap CAS.xlsm"), 'TRI', skiprows=1)


recap_cas=recap_cas[recap_cas['Confirmation']=='Acceptée']
recap_cas=recap_cas[['ENSEIGNE', 'Thème ', "Type d'offre", 'DEBUT CONSO', 'FIN CONSO', 'S début', 'Taux de dégradation', 'Unité de Besoin', 'CODE produit', 'LIBELLE']]


recap_cas['Nb de jours']= recap_cas['FIN CONSO'] - recap_cas['DEBUT CONSO']
#newdatabase(Mois)

print(data_file)


enseigne=data_file.reset_index(drop=True)['Enseigne'][:1]


df=newdatabase(5)
recap_cas["EAN 13"] = recap_cas['CODE produit'].map(bdd_produits.set_index('Code SAP')['EAN 13'])
recap_cas=recap_cas[recap_cas['ENSEIGNE']==enseigne.values[0]]



print(recap_cas)

writer = pd.ExcelWriter('performance_promo.xlsx', engine='xlsxwriter')

recap_cas.to_excel(writer, sheet_name="Magasin 1")


writer.save()
"""
for row in df.rows:
if df.loc[df['VMJ']>0]:
for n in recap_cas.rows :
    recap_cas=recap_cas.append({'Magasin' : magasin})
ajouter magasin, baseline"""


#engagement = somme des réceptions, mais idéalement somme des commandes
#Engagement=data_file['Receptions'].apply(np.sum)

"""Ventes=data_file['Ventes Unites'].apply(np.sum)

def baseline(vente, moyenne, ecarttype):
    if abs(vente - moyenne) > (2 * ecarttype):
        depollution = moyenne
    else:
        depollution = vente
        
result['baseline']=baseline()

mois=result['Date'].month()
for mois:
    for produit
result['mecanique']= data_file['Enseigne'].map(recap_cas.set_index('ENSEIGNE')["Type d'offre"])

CA = data_file['Value sales'].apply(np.sum)"""