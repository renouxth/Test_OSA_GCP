import pandas as pd
import numpy as np
import datetime as dt
import glob
import os


path = r"C:\Python27\Scripts\844182-Source-Code-10-25-2016\Source Code\Nouveau projet\crf"

all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f, encoding='ISO-8859-1', names =['Magasin', 'Produit', 'EAN 13', 'Date', 'Ventes Unites', 'Indicateur promo', 'Flag probleme', 'Stock', 'Receptions', 'Value Sales', 'Manque a gagner', 'Manque a gagner promo', 'Taux de disponibilite', 'Taux de disponibilite promo'], skiprows=1) for f in all_files)
data_file  = pd.concat(df_from_each_file, ignore_index=True)

data_file=data_file.iloc[:50000]

#data_file=pd.concat([data_file1,data_file2,data_file3])

print(data_file)

bdd_produits=pd.read_excel(pd.ExcelFile(r"C:\Python27\Scripts\844182-Source-Code-10-25-2016\Source Code\Nouveau projet\Fichier_produits_Flux.xlsx"), 'Fichier_produits_Flux', skiprows=3)

#Remplacer les valeurs NA par des valeurs nulles
data_file=data_file.fillna(0)

#Changer le format de la date

data_file['Date'] = pd.to_datetime(data_file['Date'], errors='coerce')

#for words in data_file['Magasin']: #split the sentence into individual words


if "CARREFOUR" in data_file['Magasin'].values[0]: #see if one of the words in the sentence is the word we want
    data_file['Enseigne']='Carrefour'

elif "AUCHAN" in data_file['Magasin'].values[0]: #see if one of the words in the sentence is the word we want
    data_file['Enseigne']='Auchan'

else:
    data_file['Enseigne']='Intermarche'
    

#Utiliser un format courant en France
#data_file['Date']=data_file['Date'].dt.strftime('%d/%m/%Y')


#print(data_file.loc[data_file['Flag probleme'] == 1])


#Supprimer les Gencode doublons en gardant les produits les plus récents


#bdd_produits['Date de lancement']=pd.to_datetime(bdd_produits['Date de lancement'])
bdd_produits=bdd_produits.sort_values('Date de lancement').drop_duplicates(subset='EAN 13', keep='last')



#Création de la nouvelle colonne Code SAP
data_file["Code SAP"] = data_file['EAN 13'].map(bdd_produits.set_index('EAN 13')['Code SAP'])


data_file['Mois']=data_file['Date'].dt.month


data_file['Annee']=data_file['Date'].dt.year

#data_file.loc[data_file['Gencode']==5601050033830, 'Code SAP']=123593

#Avoir les Gencodes inexistants
incorrect_values=data_file[data_file.isnull().any(axis=1)]
incorrect_values.to_csv('AnomaliesCodes.csv', sep=',')


def calculVMJ(Mois):
    #Changer l'index pour grouper selon une date
    data_file.index = pd.to_datetime(data_file.Date)
    
    
    #data_file.round({'Ventes Unites' : 6})
    data_file['Ventes Unites'] =data_file['Ventes Unites'].astype(float)
    #pd.options.display.float_format='{:,.6f}'.format
    
    result_VMJ1=data_file[(data_file['Mois']>= Mois - 3) & (data_file['Mois']<Mois)]
    
    #Créer les VMJ en groupant par Magasin, Produit, Mois
    result_VMJ1=data_file[data_file['Indicateur promo']==0]
    result_VMJ2=result_VMJ1.groupby(['Magasin','Code SAP'])
    #result_VMJ3=result_VMJ2.resample('M')
    result_VMJ4=result_VMJ2['Ventes Unites'].mean()

    #Changer la Serie en Dataframe
    result_VMJ5 = result_VMJ4.to_frame().reset_index()
    
    #Changer le nom du titre
    result_VMJ5.rename(columns={'Ventes Unites': 'VMJ'}, inplace=True)

    return result_VMJ5
    

def calculEC(Mois):
    
    result_EC1=data_file[(data_file['Mois']>= Mois - 3) & (data_file['Mois']<Mois)]
    
    #Créer les Ecarts Types en groupant par Magasin, Produit, Mois
    result_EC1=data_file[data_file['Indicateur promo']==0]
    result_EC2=result_EC1.groupby(['Magasin', 'Code SAP'])
    #result_EC3=result_EC2.resample('M').agg({'Ventes Unites':np.nanstd(axis=0)})
    #result_EC3=result_EC2.resample('M')
    
    result_EC4=result_EC2['Ventes Unites'].apply(np.std)
    
    #result_EC5=result_EC4.std(axis=0, ddof=0)

    #Changer la Serie en Dataframe
    result_EC5 = result_EC4.to_frame().reset_index()
    
    #Changer le nom du titre
    result_EC5.rename(columns={'Ventes Unites': 'Ecart Type'}, inplace=True)
    
    return result_EC5


def newdatabase(Mois):
    #Concaténer les deux tableaux
    result = pd.merge(calculVMJ(Mois), calculEC(Mois), on=['Magasin', 'Code SAP'])
    
    #Changer le format de la date
    #result['Date']=result['Date'].dt.strftime('%m-%Y')
    
    print(result)
    return result













