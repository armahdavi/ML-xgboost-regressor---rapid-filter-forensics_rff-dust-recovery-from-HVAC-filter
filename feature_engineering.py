# -*- coding: utf-8 -*-
"""
Program to run feature engineering to understand the features and their distributions

@author: alima
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

exec(open(r'C:\PhD Research\Generic Codes\notion_corrections.py').read())
df = pd.read_excel(backslash_correct(r'C:\PhD Research\Paper 1 - Extraction\Processed\natural\ml_extraction_data.xlsx'))

###########################################
### Figure 1: Plot correlational matrix ###
###########################################

## correlational matrix creation and plotting
corr_matrix = df[['runtime', 'dust_rem', 'DC 0.5-2.5 mean', 'DC 0.5-2.5 max', 'DC > 2.5 mean', 'DC > 2.5 max', 'M_t']].corr(method = 'spearman')
fig, ax = plt.subplots(figsize = (7, 7))

cmap = sns.diverging_palette(250, 10, as_cmap = True) # Custom color palette
sns.heatmap(corr_matrix, cmap = cmap, annot = True, fmt = ".2f", center = 0, linewidths = 0.5, cbar_kws = {"shrink": 0.8}) # heatmap with the full correlation matrix

## x and y label properties
plt.xlabel("Features")
plt.ylabel("Features")
plt.title("Correlation Matrix - Dust Recovery RFF")

## Saving and exhibition
plt.savefig(r'C:\PhD Research\Paper 1 - Extraction\Processed\plots\rff\correlation_matrix.jpg', format = 'jpg', dpi = 1600, bbox_inches = 'tight')
plt.show()


###########################################
### Figure 2: Key feature distributions ###
###########################################

## Creating histograms for each feature
fig, axes = plt.subplots(nrows = len(df[['runtime', 'dust_rem', 'DC 0.5-2.5 mean', 'DC 0.5-2.5 max', 'DC > 2.5 mean', 'DC > 2.5 max', 'M_t']].columns), figsize = (8, 6 * len(df[['runtime', 'dust_rem', 'DC 0.5-2.5 mean', 'DC 0.5-2.5 max', 'DC > 2.5 mean', 'DC > 2.5 max', 'M_t']].columns)))
for i, column in enumerate(df[['runtime', 'dust_rem', 'DC 0.5-2.5 mean', 'DC 0.5-2.5 max', 'DC > 2.5 mean', 'DC > 2.5 max', 'M_t']].columns):
    ax = axes[i]
    sns.histplot(data = df[['runtime', 'dust_rem', 'DC 0.5-2.5 mean', 'DC 0.5-2.5 max', 'DC > 2.5 mean', 'DC > 2.5 max', 'M_t']], x = column, ax = ax, kde = True)
    ax.set_xlabel(column)

## Adjust the spacing between subplots
plt.tight_layout()

## Saving and exhibition
plt.savefig(r'C:\PhD Research\Paper 1 - Extraction\Processed\plots\rff\distribution.jpg', format = 'jpg', dpi = 800, bbox_inches = 'tight')
plt.show()


####################################################
### Figure 2: Key feature pair plot distributions ###
####################################################

## Paired distribution of features (matrix of scatter plots)
g = sns.pairplot(df[['runtime', 'dust_rem', 'DC 0.5-2.5 mean', 'DC 0.5-2.5 max', 'DC > 2.5 mean', 'DC > 2.5 max', 'M_t']])
g.fig.suptitle('Matrix of Scatter Plots- Paired Distributions and Correlations', y = 1.03, fontsize = 36)

## Saving and exhibition
plt.savefig(r'C:\PhD Research\Paper 1 - Extraction\Processed\plots\rff\matrix_scatter.jpg', format = 'jpg', dpi = 800, bbox_inches = 'tight')
plt.show()
