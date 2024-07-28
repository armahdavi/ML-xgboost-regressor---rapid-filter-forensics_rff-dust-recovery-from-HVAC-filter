# -*- coding: utf-8 -*-
"""
Program to extract features and target for rapid filter forensics data for dust recovery prediction using PM features

@author: alima
"""

###########################################################
### Step 1: Initial Data Pipelines & Feature Extraction ###
###########################################################

import pandas as pd
exec(open('C:/PhD Research/Generic Codes/notion_corrections.py').read())

### Reading dust extraction data 
df = pd.read_excel(backslash_correct(r'C:\PhD Research\Paper 1 - Extraction\Processed\natural\natl_dataset_summary.xlsx'))
# calculating the pre-extraction filter mass
df['M_filter_pre'] = df['M_filter_post'] + df['M_filter_change_cum']
df['M_filter_bl'] = df['M_filter_pre'] - df['dustmass']


### Reading runtime data, processing it by aggregating over site and round
runtime = pd.read_stata(backslash_correct(r'C:\PhD Research\Paper 1 - Extraction\Raw\runtime_summary.dta'))
runtime['id'] = runtime['site'].astype(str).apply(lambda x: x[:-2]) + '_' + runtime['filter_type'].astype(str).apply(lambda x: x[:-2])
runtime = runtime.groupby('id', as_index = False)['runtime'].mean()
runtime['Site_N']  = runtime['id'].str.split('_').apply(lambda x: x[0]).astype(int)
runtime['Round_N'] = runtime['id'].str.split('_').apply(lambda x: x[1]).astype(int)
del runtime['id']
runtime['runtime'] = runtime['runtime'].astype(float).round(decimals = 3)

### Merging runtime with dust extraction data based on site and round numbers
df = pd.merge(df, runtime, on = ['Site_N', 'Round_N'])
df = df.dropna(subset = ['dustmass'])
df = df[['SN', 'Site_N', 'Round_N', 'runtime', 'ft', 'Cycle_N', 'dust_rem', 'dustmass', 'M_d', 'M_s', 'M_t']]


### Reading DC1700 number concentration and merging with previous df
df_dc = pd.read_excel(backslash_correct(r'C:\PhD Research\Paper 2 - PSD_QFF\Processed\dc_1700.xlsx'))
df_dc_collapse = df_dc.groupby(['site', 'round'])[['DC 0.5-2.5', 'DC > 2.5']].agg(['mean', 'max']).reset_index()
df_dc_collapse.columns = [(x + ' ' + y).strip() for x,y in df_dc_collapse.columns]
df_dc_collapse.to_excel(backslash_correct(r'C:\PhD Research\Paper 1 - Extraction\Processed\natural\dc_1700_agg.xlsx'), index = False)
df_dc_collapse.rename(columns = {'round':'Round_N', 'site':'Site_N'}, inplace = True)
df = pd.merge(df, df_dc_collapse, on = ['Site_N', 'Round_N'], how = 'left')

## Saving the processed database
df.to_excel(backslash_correct(r'C:\PhD Research\Paper 1 - Extraction\Processed\natural\ml_extraction_data.xlsx'), index = False)

