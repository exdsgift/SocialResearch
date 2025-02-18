import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings
from scipy.stats import chi2_contingency
import textwrap

warnings.filterwarnings('ignore')



def ds_filter_norway():
   dataset = pd.read_csv('ESS11/ESS11.csv', low_memory=False)
   dsNO = dataset[dataset['cntry'] == 'NO']
   dsNO = dsNO.dropna(axis= 1, how='all')
   
   colm = ['netusoft','ppltrst','pplfair','pplhlp']
   col_2 = ['netusoft','ppltrst','pplfair','pplhlp', 'gndr']
   colm2 = ['nwspol', 'netustm']
   ds1 = dsNO[colm]
   ds3 = dsNO[colm2]
   ds4 = dsNO[col_2]
   
   fig, axes = plt.subplots(2, 2, figsize = (15, 10), dpi =200)
   axes = axes.flatten()
   
   titles = ['Internet use, how often',
          'Most people can be trusted or you cant be\ntoo careful',
          'Most people try to take advantage of you,\nor try to be fair',
          'Most of the time people helpful or mostly\nlooking out for themselves']

   for i, col in enumerate(colm):
      sns.countplot(data=ds1, x = col, ax=axes[i], palette='mako')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Count')
      axes[i].grid(False)
      axes[i].set_title(f'{titles[i]} - NO')
      
      # Add percentages
      total = len(ds1[col].dropna())
      for p in axes[i].patches:
         height = p.get_height()
         percentage = f'{100 * height / total:.1f}%'
         axes[i].annotate(percentage, 
                           (p.get_x() + p.get_width() / 2., height), 
                           ha='center', va='center', 
                           xytext=(0, 5), textcoords='offset points', 
                           fontsize=10, color='black')
      
      # unique_values = ds1[col].dropna().unique()
      # legend_labels = [str(val) for val in unique_values]
      # axes[i].legend(legend_labels, title=f'{col} Values', loc='upper right')
      
   plt.tight_layout()
   plt.savefig(f'images/image8.pdf', bbox_inches='tight', dpi=300)
   plt.show()
   
   fig, axes = plt.subplots(2, 2, figsize = (15, 10), dpi =200)
   axes = axes.flatten()

   for i, col in enumerate(colm):
      sns.countplot(data=ds1, x = col, ax=axes[i], hue=ds4['gndr'])
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Count')
      axes[i].grid(False)
      axes[i].set_title(f'{titles[i]} - NO')
      
      # Add percentages
      total = len(ds1[col].dropna())
      for p in axes[i].patches:
         height = p.get_height()
         percentage = f'{100 * height / total:.1f}%'
         axes[i].annotate(percentage, 
                           (p.get_x() + p.get_width() / 2., height), 
                           ha='center', va='center', 
                           xytext=(0, 5), textcoords='offset points', 
                           fontsize=10, color='black')
      custom_labels = ['male', 'female']
      axes[i].legend(custom_labels)
      
   plt.tight_layout()
   plt.savefig(f'images/image9.pdf', bbox_inches='tight', dpi=300)
   plt.show()
   
   fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=200)
   axes = axes.flatten()

   chi2_results = {}

   for i, col in enumerate(colm):
      sns.countplot(data=ds1, x=col, ax=axes[i], hue=ds4['gndr'])
      axes[i].set_title(f'{titles[i]} - NO')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Count')
      axes[i].grid(False)
      
      # contingency table
      contingency_table = pd.crosstab(ds1[col], ds4['gndr'])
      
      # chi-squared test
      chi2, p, dof, expected = chi2_contingency(contingency_table)
      chi2_results[col] = {'chi2': chi2, 'p-value': p, 'dof': dof, 'expected': expected}
      
      # Annotate the plot
      axes[i].annotate(f'p-value: {p:.3f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10, color='black')
      
      custom_labels = ['male', 'female']
      axes[i].legend(custom_labels)

   plt.tight_layout()
   plt.savefig(f'images/image10.pdf', bbox_inches='tight', dpi=300)
   plt.show()

   for col, result in chi2_results.items():
      print(f"Column: {col}")
      print(f"Chi2: {result['chi2']}")
      print(f"P-value: {result['p-value']}")
      print(f"Degrees of Freedom: {result['dof']}")
      print(f"Expected Frequencies: \n{result['expected']}\n")
   
   
   
   
   
   
   values_remove = [6666,7777,8888,9999]
   ds3_new = ds3[~ds3['nwspol'].isin(values_remove)]
   ds3_new = ds3_new[~ds3_new['netustm'].isin(values_remove)]

   fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
   axes = axes.flatten()

   labels = ['News about politics and current affairs,\nwatching, reading or listening, in minutes',
          'Internet use, how often']

   for i, col in enumerate(colm2):
      sns.kdeplot(data=ds3_new, x=col, ax=axes[i], fill=True, palette='viridis')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Density')
      axes[i].grid(False)
      axes[i].set_title(f'{labels[i]} - NO')

      # mean and median
      mean_val = ds3_new[col].mean()
      median_val = ds3_new[col].median()

      # vertical lines
      axes[i].axvline(mean_val, color='blue', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
      axes[i].axvline(median_val, color='red', linestyle='-.', linewidth=1.5, label=f'Median: {median_val:.2f}')

      axes[i].legend(fontsize=8)

   plt.tight_layout()
   plt.savefig(f'images/image11.pdf', bbox_inches='tight', dpi=300)
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
   
   fig, axes = plt.subplots(15, 3, figsize = (15, 75), dpi=300)
   axes = axes.flatten()

   for i, col in enumerate(col2):
      sns.countplot(data=ds2, x = col, ax=axes[i], palette='pastel')
      axes[i].set_title(f'Distribution of {col} (NO)')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Count')
      axes[i].grid(False)
      
      # Add percentages
      total = len(ds2[col].dropna())
      for p in axes[i].patches:
         height = p.get_height()
         percentage = f'{100 * height / total:.1f}%'
         axes[i].annotate(percentage, 
                           (p.get_x() + p.get_width() / 2., height), 
                           ha='center', va='center', 
                           xytext=(0, 5), textcoords='offset points', 
                           fontsize=10, color='black')
      
   plt.tight_layout()
   plt.savefig(f'images/image12.pdf', bbox_inches='tight', dpi=300)
   plt.show()
   
   title = "Correlation Plot"
   
   columns = colm + colm2
   correlation_matrix = dsNO[columns].corr()
   plt.figure(figsize=(8, 6), dpi=200)
   sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
   plt.title(title)
   plt.savefig(f'images/image13.pdf', bbox_inches='tight', dpi=300)
   plt.show()

   threshold=0
   correlation_matrix2 = ds2.corr()
   #mask = (correlation_matrix.abs() < threshold)
   #filtered_correlation = correlation_matrix2.where(~mask, other=np.nan)

   plt.figure(figsize=(12, 10), dpi=200)
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
   plt.savefig(f'images/image14.pdf', bbox_inches='tight', dpi=300)
   plt.show()  
   

def correlation_plot(dataset, columns, title="Correlation Plot"):
    """
    Generate a heatmap for the correlation matrix of the specified columns in the dataset.
    """
    correlation_matrix = dataset[columns].corr()
    plt.figure(figsize=(8, 6), dpi=200)
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
   
   titles2 = ['Internet use, how often',
          'Most people can be trusted or you cant be\ntoo careful',
          'Most people try to take advantage of you,\nor try to be fair',
          'Most of the time people helpful or mostly\nlooking out for themselves']
   
   titles1 = ['News about politics and current affairs,\nwatching, reading or listening, in minutes',
          'Internet use, how often']

   fig, axes = plt.subplots(1, 2, figsize = (10, 5), dpi=200)
   axes = axes.flatten()

   for i, col in enumerate(colm2):
      sns.kdeplot(data=dsNO, x=col, ax=axes[i], fill=True, label='Norway')
      sns.kdeplot(data=dsITA, x=col, ax=axes[i], fill=True, label='Italy')
      axes[i].set_title(f'{titles1[i]}, IT vs NO')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Density')
      axes[i].legend()

   plt.tight_layout()
   plt.savefig(f'images/image15.pdf', bbox_inches='tight', dpi=300)
   plt.show()
   
   fig, axes = plt.subplots(2, 2, figsize = (10, 10), dpi=200)
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
      axes[i].set_title(f'{titles2[i]}, IT vs NO')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Count')
      axes[i].legend()
      
      # unique_values = ds1[col].dropna().unique()
      # legend_labels = [str(val) for val in unique_values]
      # axes[i].legend(legend_labels, title=f'{col} Values', loc='upper right')
      
   plt.tight_layout()
   plt.savefig(f'images/image16.pdf', bbox_inches='tight', dpi=300)
   plt.show()

   titles_compare = ['Voted last national election',
                     'Donated to or participated in political\nparty or pressure group last 12\nmonths',
                     'Worn or displayed campaign badge/sticker\nlast 12 months',
                     'Signed petition last 12 months',
                     'Boycotted certain products last 12\nmonths',
                     'Posted or shared anything about politics\nonline last 12 months',
                     'Volunteered for not-for-profit or\ncharitable organisation',
                     'Allow many/few immigrants of same\nrace/ethnic group as majority',
                     'Allow many/few immigrants of different\nrace/ethnic group from majority',
                     'Allow many/few immigrants from poorer\ncountries outside Europe']
   
   compare_list = ['vote', 'donprty', 'badge', 'sgnptit', 'bctprd', 'pstplonl', 'volunfp', 'imsmetn', 'imdfetn', 'impcntr']
   values_remove = [7, 8]
   dataset_filtered = dataset[dataset['cntry'].isin(['NO', 'IT'])]
   dataset_filtered = dataset_filtered.replace(values_remove, pd.NA)

   fig, axes = plt.subplots(4, 3, figsize=(15, 20), dpi=200)
   axes = axes.flatten()

   for i, col in enumerate(compare_list[:10]):
      data = dataset_filtered[['cntry', col]].dropna()
      data_grouped = data.groupby(['cntry', col]).size().reset_index(name='Count')
      
      total_counts = data_grouped.groupby('cntry')['Count'].transform('sum')
      data_grouped['Percentage'] = (data_grouped['Count'] / total_counts) * 100
      
      # Plot
      sns.barplot(data=data_grouped, x=col, y='Count', hue='cntry', ax=axes[i], palette='pastel')
      axes[i].set_title(f'{titles_compare[i]}. IT vs NO')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Count')
      axes[i].legend(title='Country', loc='upper right')
      
      # Add percentages
      for bar, percentage in zip(axes[i].patches, data_grouped['Percentage']):
         x = bar.get_x() + bar.get_width() / 2
         y = bar.get_height()
         axes[i].text(x, y, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)

   # Remove unused axes
   for j in range(10, len(axes)):
      fig.delaxes(axes[j])

   plt.tight_layout()
   plt.savefig(f'images/image17.pdf', bbox_inches='tight', dpi=300)
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
