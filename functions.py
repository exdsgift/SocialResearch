import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
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
   
   title = "Correlation Plot"
   
   columns = colm + colm2
   correlation_matrix = dsNO[columns].corr()
   plt.figure(figsize=(8, 6))
   sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
   plt.title(title)
   plt.show()

   threshold=0
   correlation_matrix2 = ds2.corr()
   #mask = (correlation_matrix.abs() < threshold)
   #filtered_correlation = correlation_matrix2.where(~mask, other=np.nan)

   plt.figure(figsize=(12, 10))
   sns.heatmap(
      correlation_matrix2, 
      #annot=True, 
      cmap='viridis', 
      fmt=".2f", 
      linewidths=0.5, 
      annot_kws={"size": 8}, 
      cbar_kws={"shrink": 0.8}
   )
   plt.xticks(rotation=45, ha='right', fontsize=10)
   plt.yticks(fontsize=10)
   plt.title(title, fontsize=14)
   plt.show()  
   

def correlation_plot(dataset, columns, title="Correlation Plot"):
    """
    Generate a heatmap for the correlation matrix of the specified columns in the dataset.
    """
    correlation_matrix = dataset[columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
    plt.title(title)
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



country_mapping = {
    'AL': 'Albania', 'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria',
    'CH': 'Switzerland', 'CY': 'Cyprus', 'CZ': 'Czechia', 'DE': 'Germany',
    'DK': 'Denmark', 'EE': 'Estonia', 'ES': 'Spain', 'FI': 'Finland',
    'FR': 'France', 'GB': 'United Kingdom', 'GE': 'Georgia', 'GR': 'Greece',
    'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IS': 'Iceland',
    'IL': 'Israel', 'IT': 'Italy', 'LT': 'Lithuania', 'LU': 'Luxembourg',
    'LV': 'Latvia', 'ME': 'Montenegro', 'MK': 'North Macedonia',
    'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland', 'PT': 'Portugal',
    'RO': 'Romania', 'RS': 'Serbia', 'RU': 'Russian Federation',
    'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia', 'TR': 'Turkey',
    'UA': 'Ukraine', 'XK': 'Kosovo'
}


def plot_distribution_general(column_name, fig_width=1200, fig_height=800):
   # Read the dataset
   dataframe = pd.read_csv('ESS11/ESS11.csv', low_memory=False)
   general = dataframe[['cntry', column_name]]
   
   # Replace country codes with names
   general['cntry'] = general['cntry'].replace(country_mapping)
   
   # Group by country and the specified column, then calculate distribution
   distribution = general.groupby(['cntry', column_name]).size().unstack(fill_value=0)
   
   # Dynamically define the categories based on the column's unique values
   categories = distribution.columns.tolist()
   
   # Calculate the 'Mean' and normalize the distribution
   distribution['Mean'] = (distribution * range(1, len(categories) + 1)).sum(axis=1) / distribution.sum(axis=1)
   distribution = distribution.sort_values('Mean', ascending=False)
   distribution_percentage = distribution.div(distribution.sum(axis=1), axis=0)
   
   # Create the figure
   fig = go.Figure()
   for category in categories:
      fig.add_trace(
         go.Bar(
               y=distribution_percentage.index,
               x=distribution_percentage[category],
               name=category,
               orientation='h'
         )
      )

   # Update layout
   fig.update_layout(
      title=f'{column_name} distribution by country',
      barmode='stack',
      xaxis=dict(title='Percentage', tickformat=".0%"),
      yaxis=dict(title='Country'),
      legend_title='Frequency',
      height=fig_height,  # Set height dynamically
      width=fig_width,
      margin=dict(l=100, r=50, t=50, b=50)
   )
   
   fig.show()
   
# politics_var = [
#    "polintr", "psppsgva", "actrolga", "psppipla", "cptppola", 
#    "trstprl", "trstlgl", "trstplc", "trstplt", "trstprt", 
#    "trstep", "trstun", "vote", "contplt", 
#    "donprty", "badge", "sgnptit", "pbldmna", "bctprd", 
#    "pstplonl", "volunfp", "clsprty", "prtdgcl", 
#    "lrscale", "stflife", "stfeco", "stfgov", "stfdem", 
#    "stfedu", "stfhlth", "gincdif", "freehms", "hmsfmlsh", 
#    "hmsacld", "euftf", "lrnobed", "loylead", "imsmetn", 
#    "imdfetn", "impcntr", "imbgeco", "imueclt", "imwbcnt"
# ]

def gen_plot(column_name, fig_width=1200, fig_height=800):
   
   listA = ["polintr", "psppsgva", "actrolga", "psppipla", "cptppola", "vote", "contplt",
            "donprty", "badge", "sgnptit", "pbldmna", "bctprd", "pstplonl", "volunfp",
            "clsprty", "prtdgcl", "gincdif", "freehms", "hmsfmlsh", "hmsacld", "lrnobed",
            "loylead", "imsmetn", "imdfetn", "impcntr", "imbgeco"]
   listB = ["trstprl", "trstlgl", "trstplc", "trstplt", "trstprt", "trstep", "trstun",
            "lrscale", "stflife", "stfeco", "stfgov", "stfdem", "stfedu", "stfhlth", "euftf",
            "imueclt", "imwbcnt"]

   dataframe = pd.read_csv('ESS11/ESS11.csv', low_memory=False)
   general = dataframe[['cntry', column_name]]
   general['cntry'] = general['cntry'].replace(country_mapping)
   
   if column_name in listA:
      general = general[~general[column_name].isin([6,7,8,9])]
   
   if column_name in listB:
      general = general[~general[column_name].isin([66,77,88,99])]
      
   distribution = general.groupby(['cntry', column_name]).size().unstack(fill_value=0)
   categories = distribution.columns.tolist()
   

   
   # Calculate the 'Mean' and normalize the distribution
   distribution['Mean'] = (distribution * range(1, len(categories) + 1)).sum(axis=1) / distribution.sum(axis=1)
   distribution = distribution.sort_values('Mean', ascending=False)
   distribution_percentage = distribution.div(distribution.sum(axis=1), axis=0)
   
   
   
   # Create the figure
   fig = go.Figure()
   for category in categories:
      fig.add_trace(
         go.Bar(
               y=distribution_percentage.index,
               x=distribution_percentage[category],
               name=category,
               orientation='h'
         )
      )

   # Update layout
   fig.update_layout(
      title=f'{column_name} distribution by country',
      barmode='stack',
      xaxis=dict(title='Percentage', tickformat=".0%"),
      yaxis=dict(title='Country'),
      legend_title='Frequency',
      height=fig_height,  # Set height dynamically
      width=fig_width,
      margin=dict(l=100, r=50, t=50, b=50)
   )
   
   fig.show()
