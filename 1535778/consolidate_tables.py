import os
import re
import glob
import datetime
import warnings
import pandas as pd
import numpy as np
from fuzzywuzzy import process

import sys
import warnings
sys.path.insert(0, '../') 
from utils import init_logger

# https://www.sec.gov/robots.txt
def get_standard_name(col, choices, score_cutoff=60):
    best_match, score = process.extractOne(col, choices)
    if score > score_cutoff:
        return best_match
    return col

def stopping_criterion(qtr:str)->str:
    return '{}'.format(r'Total\s*Non-Control/Non-Affiliate\s*Investments')

    # if qtr == '2023-12-31':
    #     return '{}'.format(r'Total\s*Cash\s*Equivalents')
    # return '{}'.format(r'Total\s*Investments')
    
def concat(*dfs)->list:
    final = []
    for df in dfs:
        final.extend(df.values.tolist())
    return final

def common_subheaders()->tuple:
    return tuple(map(lambda header:header.replace(' ', r'\s*'),
        ('Advertising, Public Relations and Marketing ',
        'Air Transportation',
        'Amusement and Recreation',
        'Apparel Manufacturing',
        'Building Equipment Contractors',
        'Business Support Services',
        'Chemicals',
        'Communications Equipment Manufacturing',
        'Credit Related Activities',
        'Computer Systems Design and Related Services',
        'Credit (Nondepository)',
        'Data Processing and Hosting Services',
        'Educational Support Services',
        'Electronic Component Manufacturing',
        'Equipment Leasing',
        'Facilities Support Services',
        'Grocery Stores',
        'Hospitals',
        'Insurance',
        'Lessors of Nonfinancial Licenses',
        'Management, Scientific, and Technical Consulting Services',
        'Motion Picture and Video Industries',
        'Other Information Services',
        'Other Manufacturing',
        'Other Publishing',
        'Other Real Estate Activities',
        'Other Telecommunications',
        'Plastics Manufacturing',
        'Radio and Television Broadcasting',
        'Real Estate Leasing',
        'Restaurants',
        'Retail',
        'Satellite Telecommunications',
        'Scientific Research and Development Services',
        'Texttile Furnishings Mills',
        'Traveler Arrangement',
        'Software Publishing',
        'Utility System Construction',
        'Wholesalers',
        'Wired Telecommunications Carriers',
        'Wireless Telecommunications Carriers',
        )
    ))

def standard_field_names()->tuple:
    return (
        'Portfolio Company / Type of Investment',
        'portfolio company',
        'business description',
        'type of investment',
        'investment date',
        'index rate',
        'shares/ units',
        'total rate',
        'reference rate and spread',
        'pik rate',
        'maturity date',
        'principal',
        'cost',
        'footnotes',
        'industry',
        'principal amount', # TODO change stand names for more dynamic fuzzywuzzy matching
        'fair value',
    )

def company_control_headers()->tuple:
    return tuple(map(lambda header:header.replace(' ', r'\s*'),
        (
        'Debt Investments',
        'Debt Investments (82.23%)',
        'Debt Investments (A)',
        'Debt Investments (continued)',
        'Equity Securities',
        'Equity Securities (continued)',
        'Cash and Cash Equivalents',
        )
    ))

def exceptions(cik: str)->list:
    return [
        os.path.join(cik,'2022-09-30','Schedule_of_Investments_0.csv')
    ]

def except_rows()->list:
    return (
        'As of',
        'HMS Income Fund',
        'schedule of investments'
    )
    
def strip_string(
    columns_names:list,
    standardize:bool=False
)->tuple:
    # columns = tuple(map(lambda col:re.sub(r'[^a-z]', '', str(col).lower()),columns_names))
    if standardize:
        standard_fields = standard_field_names()
        return tuple(
            re.sub(r'\s+', '_',get_standard_name(col,standard_fields)) for col in columns_names
        )
    return tuple(re.sub(r'\s+', '_',col) for col in columns_names)

def get_key_fields(
    df_cur:pd.DataFrame,
)->tuple:
    important_fields = standard_field_names() + common_subheaders()
    for idx,row in enumerate(df_cur.iterrows()):
        found = any(any(
            key in str(field).lower() 
            for key in important_fields)
                    for field in row[-1].dropna().tolist()
            )
        if found and len(set(row[-1].dropna().tolist())) >= 4:
            cols = df_cur.iloc[:idx + 1].apply(lambda row: ' '.join(row.dropna()), axis=0).tolist()
            fields = strip_string(cols,standardize=found) ,idx
            return fields
    return strip_string(df_cur.iloc[0].tolist(),standardize=found),0


# Function to extract date and convert to datetime object
def extract_date(file_path):
    # Extract date from file path (assuming date is always in 'YYYY-MM-DD' format)
    date_str = re.search(r'\d{4}-\d{2}-\d{2}', file_path).group()
    return datetime.datetime.strptime(date_str, '%Y-%m-%d')


def merge_duplicate_columns(
    df:pd.DataFrame,
    merged_pair_idxs:dict={}
)->pd.DataFrame:
    duplicate_cols = merged_pair_idxs.keys()
    flag = not merged_pair_idxs.keys()
    if flag: 
        duplicate_cols = df.columns.unique() 
    for col_name in duplicate_cols:
        mask = merged_pair_idxs.get(col_name)
        if flag:
            mask = df.columns == col_name
            merged_pair_idxs[col_name] = mask
        duplicate_data = df.loc[:, mask]
        merged_data = duplicate_data.apply(lambda row: ' '.join(set(row.dropna().astype(str))), axis=1)
        df = df.loc[:, ~mask]
        df[col_name] = merged_data
    return df.reset_index(drop=True),merged_pair_idxs

def extract_subheaders(
    df:pd.DataFrame,
    control:bool,
)->pd.DataFrame:
    col_name = 'company_control' if control else 'Type_of_Investment'
    if col_name in df.columns:
        return df
    include = df.apply(
        lambda row: re.search('|'.join(company_control_headers() if control else common_subheaders()), str(row[0]), re.IGNORECASE) is not None,
        axis=1
    )  
    
    exclude = ~df.apply(
        lambda row: row.astype(str).str.contains('total|Inc|Ltd|LLC|Holdings|LP|Co|Corporation', case=False, na=False).any(),
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
    return df


def remove_row_duplicates(row:pd.Series)->pd.Series: 
    out = []
    for v in row:
        if v in out:
            out.append(np.nan)
        else:
            out.append(v)
    return pd.Series(out)

  
def _clean(
    file_path:str,
    except_rows:str,
    merged_pair_idxs:dict={},
)->pd.DataFrame:
    df = pd.read_csv(file_path,index_col=0,na_values=[' ', ''])
    df.replace(['\u200b',None, r'^\s*$'],np.nan,regex=True,inplace=True) #':','$','%'
    df.dropna(axis=0,how='all',inplace=True)

    df = df[~df.apply(lambda row:row.astype(str).str.contains(except_rows,case=False, na=False).any(),axis=1)]
    if not merged_pair_idxs:
        important_fields,idx = get_key_fields(df)
        df.columns = important_fields
    df,merge_pair_idxs = merge_duplicate_columns(df,merged_pair_idxs=merged_pair_idxs)
    duplicate_idx = df.apply(lambda row:row[pd.to_numeric(row,errors='coerce').isna()].duplicated().sum() > 1 ,axis=1)
    clean_rows = df.loc[duplicate_idx].apply(remove_row_duplicates, axis=1).reset_index(drop=True)
    j = 0
    for i,flag in enumerate(duplicate_idx):
        if not flag:
            continue
        df.iloc[i,:] = clean_rows.loc[j,:].tolist()
        j += 1
    df.replace([''],np.nan,regex=True,inplace=True) #':','$','%'
    df.dropna(axis=1,how='all',inplace=True)
    # columns = (~df.isna()).sum(axis=0) < (6 if df.shape[0] > 10 else 2 if df.shape[0] <= 4 else 0)
    columns = [col.isdigit() or col == '' for col in df.columns.tolist()]
    df = df.drop(columns=df.columns[columns])
    return df.reset_index(drop=True),merge_pair_idxs
    
def main()->None:
    cik = os.getcwd().split(os.sep)[-1]
    qtrs = os.listdir(f'{cik}')
    ex = exceptions(cik)
    ex_rows = '|'.join(except_rows())
    for qtr in qtrs:
        if '.csv' in qtr or\
              not os.path.exists(os.path.join(cik,qtr,f'Schedule_of_Investments_0.csv')) or\
                  os.path.exists(os.path.join(cik,qtr,'output',f'{qtr}.csv')):
            logger.info(f"PROCESSED - {qtr}")
            continue
        # qtr = '2016-06-30'
        logger.info(f"PROCESSING - {qtr}")

        index_list_sum = i = 0
        soi_files = sorted([
            os.path.join(cik,qtr,file) 
            for file in os.listdir(os.path.join(cik,qtr))
            if '.csv' in file
        ],key=lambda f: int(f.split('_')[-1].split('.')[0]))
        soi_files = [f for f in soi_files if f not in ex]

        df,merged_pair_idxs = _clean(soi_files[i],except_rows=ex_rows,merged_pair_idxs={})
        index_list = df.apply(
            lambda row:row.astype(str).str.contains(stopping_criterion(qtr), case=False, na=False).any(),
            axis=1
        )
        index_list_sum = index_list.sum()
        dfs = [df]     
        i += 1
        while index_list_sum == 0:
            logger.info(soi_files[i])
            df,merged_pair_idxs = _clean(soi_files[i],except_rows=ex_rows,merged_pair_idxs=merged_pair_idxs)
            dfs.append(df)
            index_list = df.apply(
                lambda row:row.astype(str).str.contains(stopping_criterion(qtr), case=False, na=False).any(),
                axis=1
            )
            index_list_sum = index_list.sum()
            i += 1
            
        date_final = dfs[0]
        if len(dfs) > 1:
            date_final = pd.concat(dfs,axis=0,ignore_index=True)#pd.DataFrame(concat(*dfs))
        # date_final = extract_subheaders(date_final,control=True)
        # date_final = extract_subheaders(date_final,control=False)

        date_final['qtr'] = qtr.split(os.sep)[-1]
        if not os.path.exists(os.path.join(cik,qtr,'output')):
            os.makedirs(os.path.join(cik,qtr,'output'))
        columns_to_drop = date_final.notna().sum() <= 2
        date_final.drop(columns=columns_to_drop[columns_to_drop].index)
        date_final.to_csv(os.path.join(cik,qtr,'output',f'{qtr}.csv'),index=False)
        # break
    
    # Use glob to find files
    files = sorted(glob.glob(os.path.join(cik,'*','output','*.csv')), key=extract_date)
    single_truth = pd.concat([
        pd.read_csv(df) for df in files
    ],axis=0,ignore_index=True)
    single_truth.drop(columns=single_truth.columns[['Unnamed' in col for col in single_truth.columns]],inplace=True)
    single_truth.to_csv(f'{cik}_soi_table.csv',index=False)
    
if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    cik = os.path.basename(__file__).split('.py')[0]
    if os.path.exists(f"logs/{cik}.log"):
        os.remove(f"logs/{cik}.log")
    logger = init_logger(cik)
    logger.info(cik)
    
    main()
    
