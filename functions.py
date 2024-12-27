import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')



def ds_filter_norway():
   dataset = pd.read_csv('ESS11/ESS11.csv', low_memory=False)
   dsNO = dataset[dataset['cntry'] == 'NO']
   dsNO = dsNO.dropna(axis= 1, how='all')
   
   colm = ['netusoft','ppltrst','pplfair','pplhlp']
   colm2 = ['nwspol', 'netustm']
   
   ds1 = dsNO[colm]
   ds3 = dsNO[colm2]
   
   fig, axes = plt.subplots(2, 2, figsize = (10, 10))
   axes = axes.flatten()

   for i, col in enumerate(colm):
      sns.countplot(data=ds1, x = col, ax=axes[i], palette='mako')
      axes[i].set_title(f'Distribution of {col} (NO)')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Count')
      
      # unique_values = ds1[col].dropna().unique()
      # legend_labels = [str(val) for val in unique_values]
      # axes[i].legend(legend_labels, title=f'{col} Values', loc='upper right')
      
   plt.tight_layout()
   plt.show()
   
   values_remove = [6666,7777,8888,9999]
   ds3_new = ds3[~ds3['nwspol'].isin(values_remove)]
   ds3_new = ds3_new[~ds3_new['netustm'].isin(values_remove)]

   fig, axes = plt.subplots(1, 2, figsize = (10, 5))
   axes = axes.flatten()

   for i, col in enumerate(colm2):
      sns.kdeplot(data=ds3_new, x=col, ax=axes[i], fill=True, palette='mako')
      axes[i].set_title(f'Distribution of {col} (NO)')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Density')

   plt.tight_layout()
   plt.show()
   
   col2 = [
    "polintr", "psppsgva", "actrolga", "psppipla", "cptppola", 
    "trstprl", "trstlgl", "trstplc", "trstplt", "trstprt", 
    "trstep", "trstun", "vote", "prtvtcno", "contplt", 
    "donprty", "badge", "sgnptit", "pbldmna", "bctprd", 
    "pstplonl", "volunfp", "clsprty", "prtclcno", "prtdgcl", 
    "lrscale", "stflife", "stfeco", "stfgov", "stfdem", 
    "stfedu", "stfhlth", "gincdif", "freehms", "hmsfmlsh", 
    "hmsacld", "euftf", "lrnobed", "loylead", "imsmetn", 
    "imdfetn", "impcntr", "imbgeco", "imueclt", "imwbcnt"
   ]
   ds2 = dsNO[col2]
   
   fig, axes = plt.subplots(15, 3, figsize = (15, 75))
   axes = axes.flatten()

   for i, col in enumerate(col2):
      sns.countplot(data=ds2, x = col, ax=axes[i], palette='mako')
      axes[i].set_title(f'Distribution of {col} (NO)')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Count')
      
   plt.tight_layout()
   plt.show()
   

def compare_graphs():
   dataset = pd.read_csv('ESS11/ESS11.csv', low_memory=False)
   
   dsNO = dataset[dataset['cntry'] == 'NO'].dropna(axis= 1, how='all')
   dsITA = dataset[dataset['cntry'] == 'IT'].dropna(axis= 1, how='all')
   
   colm1 = ['netusoft','ppltrst','pplfair','pplhlp']
   colm2 = ['nwspol', 'netustm']
   
   dsNO = dsNO[colm2]
   dsITA = dsITA[colm2]

   values_remove = [6666,7777,8888,9999]
   
   dsNO = dsNO[~dsNO['nwspol'].isin(values_remove)]
   dsNO = dsNO[~dsNO['netustm'].isin(values_remove)]
   
   dsITA= dsITA[~dsITA['nwspol'].isin(values_remove)]
   dsITA = dsITA[~dsITA['netustm'].isin(values_remove)]  

   fig, axes = plt.subplots(1, 2, figsize = (10, 5))
   axes = axes.flatten()

   for i, col in enumerate(colm2):
      sns.kdeplot(data=dsNO, x=col, ax=axes[i], fill=True, label='Norway')
      sns.kdeplot(data=dsITA, x=col, ax=axes[i], fill=True, label='Italy')
      axes[i].set_title(f'Distribution of {col}')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Density')
      axes[i].legend()

   plt.tight_layout()
   plt.show()
   
   fig, axes = plt.subplots(2, 2, figsize = (10, 10))
   axes = axes.flatten()
   dsNO = dataset[dataset['cntry'] == 'NO'].dropna(axis= 1, how='all')
   dsITA = dataset[dataset['cntry'] == 'IT'].dropna(axis= 1, how='all')

   values_remove = [77,88,99]
   
   dsNO = dsNO[~dsNO['ppltrst'].isin(values_remove)]
   dsNO = dsNO[~dsNO['pplfair'].isin(values_remove)]
   dsNO = dsNO[~dsNO['pplhlp'].isin(values_remove)]
   
   dsITA = dsITA[~dsITA['ppltrst'].isin(values_remove)]
   dsITA = dsITA[~dsITA['pplfair'].isin(values_remove)]
   dsITA = dsITA[~dsITA['pplhlp'].isin(values_remove)]

   for i, col in enumerate(colm1):
      sns.kdeplot(data=dsNO, x = col, ax=axes[i], fill=True, label = 'Norway')
      sns.kdeplot(data=dsITA, x = col, ax=axes[i], fill=True, label = 'Italy')
      axes[i].set_title(f'Distribution of {col}')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Count')
      axes[i].legend()
      
      # unique_values = ds1[col].dropna().unique()
      # legend_labels = [str(val) for val in unique_values]
      # axes[i].legend(legend_labels, title=f'{col} Values', loc='upper right')
      
   plt.tight_layout()
   plt.show()