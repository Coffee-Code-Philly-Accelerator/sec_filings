import os
import re
import glob
import datetime
import pandas as pd
import numpy as np
import sys
import warnings
sys.path.insert(0, '../') 

from utils import init_logger 

def stopping_criterion(
    search_string:str='total investments'
)->str:
    # Regular expression to ignore whitespace and case
    regex_pattern = search_string.replace(' ', r'\s*')
    return '{}|{}'.format(regex_pattern,'Invesmtents')

def concat(*dfs)->list:
    final = []
    for df in dfs:
        final.extend(df.values.tolist())
    return final

def common_subheaders()->tuple:
    return tuple(map(lambda header:header.replace(' ', r'\s*'),
        ('senior secured loans',
        'first lien',
        'second lien',
        'senior secured bonds',
        'subordinated debt',
        'equity/other',
        'collateralized securities',
        'preferred equity—',
        'Equity/Warrants',
        'unsecured debt',
        'senior secured notes',
        'warrants',
        'total senior secured first lien term loans')
    ))

def standard_field_names()->tuple:
    return (
        'portfolio',
        'footnotes',
        'industry',
        'rate',
        'floor',
        'maturity',
        'principal amount', # TODO change stand names for more dynamic fuzzywuzzy matching
        'cost',
        'value',
        'investment',
        'date',
        'subheaders',
        'number of shares'
    )

def company_control_headers()->tuple:
    return tuple(map(lambda header:header.replace(' ', r'\s*'),
        (
        'control investments',
        'affiliate investments',
        'non-control/non-affilate investments',
        'Non-Controlled/Non-Affiliated  Investments:',
        'Affiliated  Investments:',
        'Non-Controlled/Non-Affiliated  Investments  :',
        'Affiliated  Investments  :',
        'Non-controlled/Non-affiliated Investments',
        'Affiliated Investments',
        )
    ))

# Function to extract date and convert to datetime object
def extract_date(file_path):
    # Extract date from file path (assuming date is always in 'YYYY-MM-DD' format)
    date_str = re.search(r'\d{4}-\d{2}-\d{2}', file_path).group()
    return datetime.datetime.strptime(date_str, '%Y-%m-%d')

def remove_row_duplicates(row):
    seen = set()
    return pd.Series([x if x not in seen and not seen.add(x) else None for x in row])

def shift_part(row,left,after=3):
    # Keeping the first three values unchanged
    first_part = row[:after]
    # Shifting the remaining part of the row to the left by 1
    shifted_part = row[after:].shift(-1 if left else 1)
    # Concatenating the two parts
    return pd.concat([first_part, shifted_part])



def merge_duplicate_columns(
    df:pd.DataFrame,
)->pd.DataFrame:
    duplicate_cols = df.columns[df.columns.duplicated(keep=False)]
    for col_name in duplicate_cols.unique():
        duplicate_data = df.loc[:, df.columns == col_name]
        merged_data = duplicate_data.apply(lambda row: ' '.join(set(row.dropna().astype(str))), axis=1)
        df = df.loc[:, df.columns != col_name]
        df[col_name] = merged_data
    return df

def extract_subheaders(
    df:pd.DataFrame,
    control:bool,
)->pd.DataFrame:
    col_name = 'company_control' if control else 'TypeofInvestment'
    if col_name in df.columns:
        return df
    include = df.apply(
        lambda row: re.search('|'.join(company_control_headers() if control else common_subheaders()), str(row[0]), re.IGNORECASE) is not None,
        axis=1
    )  
    
    exclude = ~df.apply(
        lambda row: row.astype(str).str.contains('total', case=False, na=False).any(),
        axis=1
    )
    idx = df[include & exclude].index.tolist()
    df[col_name] = None
    if not idx:
        return df

    prev_header = subheader = None
    df.loc[idx[-1]:,col_name] = df.iloc[idx[-1],1] if isinstance(df.iloc[idx[-1],0],float)  else df.iloc[idx[-1],0]
    for j,i in enumerate(idx[:-1]):
        prev_header = subheader
        subheader = df.iloc[i,1] if isinstance(df.iloc[i,0],float)  else df.iloc[i,0]
        df.loc[idx[j]:idx[j+1],col_name] = subheader if subheader != '' else prev_header
    # df.drop(idx,axis=0,inplace=True,errors='ignore') 
    return df


def clean_bbdc(
    file_path:str
)->pd.DataFrame:
    df = pd.read_csv(file_path,index_col=0)
    df.dropna(axis=1,how='all',inplace=True)
    df.dropna(axis=0,how='all',inplace=True)
    df.replace(['\u200b','%',],np.nan,inplace=True) #':','$','%'

    if not '2012-06-30' in file_path:
        df[df.apply(lambda row: row.astype(str).str.contains('TRIANGLE CAPITAL CORPORATION').any(), axis=1)] = ''
        regex_pattern = '|'.join(map(re.escape, standard_field_names()))
        mask = df.iloc[:,0].str.contains(regex_pattern, case=False, na=False)
        df.iloc[0] = df.iloc[:mask.idxmax()].astype(str).agg(' '.join)
        df.reset_index(drop=True,inplace=True)
        df.drop(axis=0, index=1,inplace=True)

    for i in range(df.shape[0]):
        df.iloc[i] = [np.nan if  item in ['None None','nan nan'] else str(item).replace('nan','').replace('None','') for item in df.iloc[i]]
        
    if '2010-09-30/Schedule_of_Investments_0.csv' in file_path:
        df.iloc[0] = shift_part(df.iloc[0],left=True)
    if '2010-09-30/Schedule_of_Investments_1.csv' in file_path:
        df.iloc[0] = shift_part(df.iloc[0],left=False,after=8)

        
    df.columns = df.iloc[0].astype(str)
    df = merge_duplicate_columns(df)
    df = df.apply(remove_row_duplicates, axis=1)
    non_empty_counts = df.applymap(lambda x: x != '' and pd.notna(x)).sum()
    return df.loc[:, non_empty_counts > 3]



def main()->None:
    qtrs = os.listdir(os.getcwd())
    for qtr in qtrs:
        if '.csv' in qtr or not os.path.exists(os.path.join(qtr,f'Schedule_of_Investments_0.csv')):
            continue
        dfs = []
        index_list_sum = i = 0
        # qtr = '2010-09-30'
        while index_list_sum == 0:
            qtr_file = os.path.join(qtr,f'Schedule_of_Investments_{i}.csv')
            logger.info(qtr_file)
            df = clean_bbdc(qtr_file)
            dfs.append(df)
            index_list = df.apply(
                lambda row:row.astype(str).str.contains(stopping_criterion(), case=False, na=False).any(),
                axis=1
            )
            index_list_sum = index_list.sum()
            i += 1
        
        date_final = pd.DataFrame(concat(*dfs))
        date_final = extract_subheaders(date_final,control=True)

        date_final['qtr'] = qtr.split('\\')[-1]
        for i in range(3):
            date_final[date_final.columns[i]].fillna(method='ffill',inplace=True)
        if not os.path.exists(os.path.join(qtr,'output')):
            os.makedirs(os.path.join(qtr,'output'))
        date_final.to_csv(os.path.join(qtr,'output',f'{qtr}.csv'),index=False)
        # break
    # Use glob to find files
    files = sorted(glob.glob(f'*/output/*.csv'), key=extract_date)
    single_truth = pd.concat([
        pd.read_csv(df) for df in files
    ],axis=0,ignore_index=True)
    single_truth.to_csv(f'{cik}_soi_table.csv',index=False)
    
if __name__ == "__main__":
    # Suppress future warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    cik = 1379785
    if os.path.exists(f"../logs/{cik}.log"):
        os.remove(f"../logs/{cik}.log")
    logger = init_logger(cik)
    main()