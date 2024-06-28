import os
import glob
import re
import warnings
import datetime
import pandas as pd 
import numpy as np
import sys
sys.path.insert(0, '../') 

from utils import init_logger
import warnings

def date_exceptions()->tuple:
    return (
        datetime.datetime(2017, 3, 31),
        datetime.datetime(2017, 6, 30),
        datetime.datetime(2017, 9, 30),
        datetime.datetime(2017, 12, 31),
        datetime.datetime(2018, 3, 31),
        datetime.datetime(2018, 6, 30),
        datetime.datetime(2018, 9, 30),
        datetime.datetime(2018, 12, 31),
    )

# TODO fix stopping criteria
def stopping_criterion(
    qtr:str,
    date_format:str='%Y-%m-%d'
)->str:
    qtr = datetime.datetime.strptime(qtr, date_format)
    if qtr <= datetime.datetime(2010, 3, 31) or qtr >= datetime.datetime(2017,3, 31):
        return '{}|{}'.format(r'^(?!.*\bSubtotal\b)(?=.*\btotal investments\b).+$',r'(?<!\S)Total\s+Portfolio\s+Investments')
    return '{}|{}'.format(r'^(?!.*\bSubtotal\b)(?=.*\btotal investments\b).+$',r'^(?!.*\bSubtotal\b)(?=.*\bTotal Idle Fund Investments).+$')


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
        'total senior secured first lien term loans',
        'secured debt')
    ))

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

def extract_subheaders(
    df:pd.DataFrame,
    control:bool,
)->pd.DataFrame:
    col_name = 'company_control' if control else 'TypeofInvestment'
    if col_name in df.columns:
        return df
    include = df.apply(
        lambda row: re.search('|'.join(company_control_headers() if control else common_subheaders()), str(row[0]), re.IGNORECASE) is not None ,
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

def remove_row_duplicates(row):
    seen = set()
    return pd.Series([x if x not in seen and not seen.add(x) else None for x in row])

def concat(*dfs)->list:
    final = []
    for df in dfs:
        final.extend(df.values.tolist())
    return final

def _clean_qtr(
    file_path:str
)->pd.DataFrame:
    df = pd.read_csv(file_path,index_col=0)
    df.replace(['\u200b','',')',':','$','%',0],np.nan,inplace=True) #':','$','%'
    df.dropna(axis=1,how='all',inplace=True)
    df.dropna(axis=0,how='all',inplace=True)
    
    cols = df.iloc[0].str.replace('[^a-zA-Z]', '', regex=True)
    if ((cols == '') + (cols == 'nan') + (cols == np.nan) + (cols == 'NaN')).sum() > 1:
        df = df.apply(remove_row_duplicates, axis=1)
    else:
        df.columns = cols
    df = merge_duplicate_columns(df.reset_index(drop=True))
    df.replace(['\u200b','',')',':','$','%',0],np.nan,inplace=True) #':','$','%'
    if ((cols == '') + (cols == 'nan') + (cols == np.nan) + (cols == 'NaN')).sum() > 1:
        df.dropna(axis=1,how='all',inplace=True)
        df.dropna(axis=0,how='all',inplace=True)
    df.drop(columns=df.columns[pd.isna(df.columns)].tolist() + [col for col in df.columns if col == ''],axis=1,inplace=True)
    return df




def main()->None:
    qtrs = os.listdir(os.getcwd())
    for qtr in qtrs:
        if '.csv' in qtr or '.py' in qtr:
            continue
        # qtr = '2013-06-30'
        logger.info(qtr)
        index_list_sum = 0
        i = 1
        df = _clean_qtr( os.path.join(qtr,f'Schedule_of_Investments_{i - 1}.csv'))
        col = df.columns.tolist()
        dfs = [df]
        while index_list_sum == 0:
            qtr_file = os.path.join(qtr,f'Schedule_of_Investments_{i}.csv')
            logger.info(qtr_file)
            df = _clean_qtr(qtr_file)
            df.columns = col
            dfs.append(df)
            index_list = df.apply(
                lambda row:row.astype(str).str.contains(stopping_criterion(qtr), case=False, na=False).any(),
                axis=1
            )
            index_list_sum = index_list.sum()
            i += 1
        date_final = pd.concat(dfs,axis=0,ignore_index=True)#pd.DataFrame(concat(*dfs))   
        # Create a dictionary for renaming columns
        rename_dict = {col: 'portfoliocompany' for col in df.columns if 'portfolio' in col.lower()}
        
        date_final.rename(columns=rename_dict, inplace=True)
        date_final = date_final[['portfoliocompany']+[col for col in date_final.columns if col != 'portfoliocompany' ]]
        date_final = extract_subheaders(date_final,control=True)
        date_final = extract_subheaders(date_final,control=False)
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
    cik = 1396440
    if os.path.exists(f"../logs/{cik}.log"):
        os.remove(f"../logs/{cik}.log")
    logger = init_logger(cik)
    main()