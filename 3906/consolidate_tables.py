
import os
import re
import glob
import json
import datetime
import warnings
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from bs4 import BeautifulSoup
from dateutil.parser import parse

import sys
import warnings
sys.path.insert(0, '../') 
from utils import init_logger

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
        'portfolio_company',
        # 'Portfolio Company /Principal Business',
        'investment_/interest_rate_/maturity',
        'percentage__interest/__shares',
        # 'Principal',
        'cost',
        'value',
        'short-term_investments',
        'percentage_ownership',
        'percent_of_class_held',
        # 'Investment',
        'percent_of_interests_held',
        # 'Industry',
        'spread_above_index',
        'aquisition_date',
        # 'Maturity',
        # 'Principal/Shares',
        # 'Investment Type',
        'of_Net_Assets',
        # 'business description',
        # 'type of investment',
        # 'investment date',
        'reference_rate_and_spread',
        'pik_rate',
        # 'maturity date',
        # 'cost',
        'footnotes',
        # 'industry',
        # 'principal amount',
        # 'fair value',
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


def except_rows()->tuple:
    return (
        'Private Finance',
        'December 31, 2001',
    )

# https://www.sec.gov/robots.txt
def get_standard_name(col, choices, score_cutoff=60):
    best_match, score = process.extractOne(col, choices)
    if score > score_cutoff:
        return best_match
    return col


def stopping_criterion(qtr:str)->str:
    if qtr == '2004-12-31':
        return '{}'.format(r'Total')
    return '{}'.format(r'Total *private *finance')


def concat(*dfs)->list:
    final = []
    for df in dfs:
        final.extend(df.values.tolist())
    return final

    
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
        if found and len(set(row[-1].dropna().tolist())) >= 6:
            cols = df_cur.iloc[:idx + 1].apply(lambda row: ' '.join(row.dropna()), axis=0).tolist()
            fields = strip_string(cols,standardize=found) 
            return fields
    return strip_string(df_cur.iloc[0].tolist())

def strip_string(
    columns_names:list,
    standardize:bool=False,
)->tuple:
    # columns = tuple(map(lambda col:re.sub(r'[^a-z]', '', str(col).lower()),columns_names))
    if standardize:
        standard_fields = standard_field_names()
        return tuple(
            re.sub(r'[^a-zA-Z]', '_',get_standard_name(str(col).strip().lower(),standard_fields)) for col in columns_names
        )
    return tuple(re.sub(r'[^a-zA-Z]', '_',str(col).strip().lower()) for col in columns_names)


# Function to extract date and convert to datetime object
def extract_date(file_path):
    # Extract date from file path (assuming date is always in 'YYYY-MM-DD' format)
    date_str = re.search(r'\d{4}-\d{2}-\d{2}', file_path).group()
    return datetime.datetime.strptime(date_str, '%Y-%m-%d')

def remove_datetime(x):
    try:
        parse(x)
        return np.nan
    except Exception as e:
        return x

def merge_duplicate_columns(
    df:pd.DataFrame,
    merged_pair_idxs:dict={}
)->pd.DataFrame:
    duplicate_cols = merged_pair_idxs.keys()
    flag = not merged_pair_idxs.keys()
    if flag: 
        duplicate_cols = df.columns.unique() 
    for col_name in duplicate_cols:
        # display(col_name)
        mask = merged_pair_idxs.get(col_name)
        if flag:
            mask = df.columns == col_name
            merged_pair_idxs[col_name] = mask
        duplicate_data = df.loc[:, mask]
        merged_data = duplicate_data.apply(lambda row: ' '.join(set(row.dropna().astype(str))), axis=1)
        df = df.loc[:, ~mask]
        df[col_name] = merged_data
        # display(df)
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
        if v in out and not str(v).replace('$','').isnumeric():
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
    df.replace(['Private Finance','(unaudited)','in thousands, except number of shares'],'',regex=True,inplace=True)
    df.replace(to_replace=r'[\[\]\(\){},$%˄\xa0\u200b]"', value='', regex=True,inplace=True)
    df = df.applymap(remove_datetime)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  
    df.replace(to_replace=r'\(|\)', value='', regex=True,inplace=True)
    df.replace(to_replace='', value=np.nan, regex=True,inplace=True)

    df.dropna(axis=0,how='all',inplace=True)

    # df = df[~df.apply(lambda row:row.astype(str).str.contains(except_rows,case=False, na=False).any(),axis=1)]
    if not df.apply(lambda col: col.astype(str).str.contains(r'total_investments', case=False, regex=True)).any().any() and df.shape[0] < 3:
        return pd.DataFrame(), merged_pair_idxs

    if not merged_pair_idxs:
        important_fields = strip_string(get_header_rows(df),standardize=True)#get_key_fields(df)
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
    columns = [col.isdigit() for col in df.columns]
    df = df.drop(columns=df.columns[columns])
    return df.reset_index(drop=True),merge_pair_idxs



def get_header_rows(
    df_cur:pd.DataFrame,
)->tuple:
    for idx,row in df_cur.reset_index().iterrows():
        found = any(str(v).replace("$",'').replace("%",'').isnumeric() for v in row)
        if found and  len(set(row.dropna().tolist())) >= 3:     
            out = df_cur.iloc[:idx + 1,:].apply(
                lambda row: ' '.join(
                    row[row.notna()].astype(str).values
            ), axis=0)
  
            return out
    
    return strip_string(df_cur.iloc[0].tolist())

def to_parse()->tuple:
    return (
        '2000-06-30',
        '2000-09-30',
        '2000-12-31',
        '2001-03-31',
        '2001-06-30',
    )


def md_parse(
    xml_file:str,
    start_str:str='CONSOLIDATED STATEMENT OF INVESTMENTS',
    end_str:str='Total private finance',
    columns:list=['portfolio_company','investment','cost','fair_value']
)->pd.DataFrame:
    # Load HTML content
    with open(xml_file, 'r') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    # If the content is inside <pre>, get the text within it
    pre_text = soup.find('pre').get_text()
    # logger.info(pre_text)
    start = pre_text.split(start_str)[-1]

    # start = start[:len(start)]
    end = start.split(end_str)[0]
    # logger.info(end)
    lines = end.split('\n')
    data = []
    num_cols = len(lines[0].split())
    for line in lines:
        line = line.strip('< >-')
        line = re.sub(r'[<>]', '', line)  # Replace occurrences of < or > with nothing
        row = [np.nan]*num_cols
        values = re.split(r'\s{4,}', line)
        if len(values) < 4:
            continue
        row[:len(values)] = values[:4]
        data.append(row)   
    df = pd.DataFrame(data,columns=columns).dropna(axis=1,how='all')
    return df

def main()->None:
    cik = os.getcwd().split(os.sep)[-1]
    qtrs = os.listdir(f'{cik}')
    # ex = exceptions()
    if os.path.exists("manual_mask.json"):
        with open("manual_mask.json", 'r') as f:
            ex = json.load(f)
        for e in ex:
            ex[e] = {c:np.array(mask) for c,mask in ex[e].items()}
    
    ex_rows = '|'.join(except_rows())
    md = to_parse()
    for qtr in qtrs:
        if '.csv' in qtr or\
              not os.path.exists(os.path.join(cik,qtr,f'Schedule_of_Investments_0.csv')) and qtr not in md or\
                  os.path.exists(os.path.join(cik,qtr,'output',f'{qtr}.csv')):
            continue
        if qtr in md:
            logger.info(f"PROCESSING MARKDOWN - {qtr}")
            f = os.listdir(os.path.join(cik,qtr))
            f.remove('output')
            df = md_parse(os.path.join(cik,qtr,f[0]))
            df['qtr'] = qtr.split(os.sep)[-1]
            if not os.path.exists(os.path.join(cik,qtr,'output')):
                os.mkdir(os.path.join(cik,qtr,'output'))
            df.to_csv(f"{os.path.join(cik,qtr,'output',qtr)}.csv",index=False)
            continue

        # qtr = '2002-06-30'
        logger.info(f"PROCESSING - {qtr}")
        index_list_sum = i = 0
        soi_files = sorted([
            os.path.join(cik,qtr,file) 
            for file in os.listdir(os.path.join(cik,qtr))
            if '.csv' in file and 'output' not in file
        ],key=lambda f: int(f.split('_')[-1].split('.')[0]))
        merged_pair_idxs = ex.get(soi_files[i],{})
        df,merged_pair_idxs = _clean(soi_files[i],except_rows=ex_rows,merged_pair_idxs=merged_pair_idxs)

        index_list = df.apply(
            lambda row:row.astype(str).str.contains(stopping_criterion(qtr), case=False, na=False).any(),
            axis=1
        )
        index_list_sum = index_list.sum()
        dfs = [df]     
        i += 1

        while index_list_sum == 0:
            logger.info(soi_files[i])
            merged_pair_idxs = ex.get(soi_files[i],{})

            # display(merged_pair_idxs)
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
            date_final = pd.concat(dfs,axis=0,ignore_index=True)
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
    logger.info(f"COMPLETED - {cik}")
    
def exceptions()->dict:
    return {
        '3906/2004-12-31/Schedule_of_Investments_2.csv':{
            'portfolio_company':np.array([True]+[False]*24),
            'investment':np.array([False,True]+[False]*23),
            'interest':np.array([False]*3+[True]+[False]*21),
            'other':np.array([False]*4+[True]+[False]*20),
            'value_prev':np.array([False]*6+[True]+[False]*18),
            'additions':np.array([False]*9+[True]+[False]*15),
            'reductions':np.array([False]*12+[True]+[False]*12),
            'value_cur':np.array([False]*15+[True]+[False]*9),
        },
        '3906/2004-12-31/Schedule_of_Investments_0.csv':{
            'portfolio_company':np.array([True]+[False]*23),
            'investment':np.array([False,True]+[False]*22),
            'interest':np.array([False]*3+[True]+[False]*20),
            'other':np.array([False]*4+[True]+[False]*19),
            'value_prev':np.array([False]*6+[True]+[False]*17),
            'additions':np.array([False]*9+[True]+[False]*14),
            'reductions':np.array([False]*12+[True]+[False]*11),
            'value_cur':np.array([False]*15+[True]+[False]*8),
        },
        '3906/2004-12-31/Schedule_of_Investments_1.csv':{
            'portfolio_company':np.array([True]+[False]*23),
            'investment':np.array([False,True]+[False]*22),
            'interest':np.array([False]*3+[True]+[False]*20),
            'other':np.array([False]*4+[True]+[False]*19),
            'value_prev':np.array([False]*6+[True]+[False]*17),
            'additions':np.array([False]*9+[True]+[False]*14),
            'reductions':np.array([False]*12+[True]+[False]*11),
            'value_cur':np.array([False]*15+[True]+[False]*8),
        },
        '3906/2001-09-30/Schedule_of_Investments_0.csv':{
            'portfolio_company':np.array([True]+[False]*11),
            'investment':np.array([False]*2+[True]+[False]*9),
            'cost':np.array([False]*4+[True]+[False]*7),
            'value':np.array([False]*7+[True]+[False]*4),
        },
        '3906/2001-09-30/Schedule_of_Investments_2.csv':{
            'portfolio_company':np.array([True]+[False]*11),
            'investment':np.array([False]*2+[True]+[False]*9),
            'cost':np.array([False]*4+[True]+[False]*7),
            'value':np.array([False]*7+[True]+[False]*4),
        },
        '3906/2001-09-30/Schedule_of_Investments_4.csv':{
            'portfolio_company':np.array([True]+[False]*11),
            'investment':np.array([False]*2+[True]+[False]*9),
            'cost':np.array([False]*4+[True]+[False]*7),
            'value':np.array([False]*7+[True]+[False]*4),
        },
        '3906/2001-09-30/Schedule_of_Investments_6.csv':{
            'portfolio_company':np.array([True]+[False]*11),
            'investment':np.array([False]*2+[True]+[False]*9),
            'cost':np.array([False]*4+[True]+[False]*7),
            'value':np.array([False]*7+[True]+[False]*4),
        },
        '3906/2001-09-30/Schedule_of_Investments_8.csv':{
            'portfolio_company':np.array([True]+[False]*11),
            'investment':np.array([False]*2+[True]+[False]*9),
            'cost':np.array([False]*4+[True]+[False]*7),
            'value':np.array([False]*7+[True]+[False]*4),
        },
        '3906/2001-09-30/Schedule_of_Investments_10.csv':{
            'portfolio_company':np.array([True]+[False]*11),
            'investment':np.array([False]*2+[True]+[False]*9),
            'cost':np.array([False]*4+[True]+[False]*7),
            'value':np.array([False]*7+[True]+[False]*4),
        },
        '3906/2001-09-30/Schedule_of_Investments_12.csv':{
            'portfolio_company':np.array([True]+[False]*10),
            'investment':np.array([False]+[True]+[False]*9),
            'cost':np.array([False]*3+[True]+[False]*7),
            'value':np.array([False]*6+[True]+[False]*4),
        },
    }

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    program = "consolidate_tables"#os.path.basename(__file__).split('.py')[0]
    if os.path.exists(f"logs/{program}.log"):
        os.remove(f"logs/{program}.log")
    logger = init_logger(program)
    logger.info(program)
    
    main()
