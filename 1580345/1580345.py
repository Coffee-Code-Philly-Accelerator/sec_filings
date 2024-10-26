import os
import re
import glob
import warnings
import datetime
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from fuzzywuzzy import process
import sys
sys.path.insert(0, '../') 
from utils import init_logger

# https://www.sec.gov/robots.txt
def get_standard_name(col, choices, score_cutoff=60):
    best_match, score = process.extractOne(col, choices)
    if score > score_cutoff:
        return best_match
    return col

def stopping_criterion(qtr:str)->str:
    if qtr == '2023-12-31':
        return '{}'.format(r'Total\s*Cash\s*Equivalents')
    return '{}'.format(r'Total\s*Investments')


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
        'portfolio',
        'footnotes',
        'industry',
        'rate',
        'floor',
        'maturity',
        'principal amount', # TODO change stand names for more dynamic fuzzywuzzy matching
        'cost',
        'fair value',
        'investment',
        'date',
        'subheaders',
        'number of shares',
        'of net assets',
        'type',
        'effective yield',
        'share units',
        'Percent of Case and Investments',
        '\\% of Portfolio',
        'issuer'
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
    df_cur:pd.DataFrame
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

def remove_regex(qtr:str)->tuple:
    if qtr == "2019-12-31":
        return re.compile(r"TRIPLEPOINT VENTURE GROWTH BDC CORP\. AND SUBSIDIARIES\s+CONSOLIDATED SCHEDULE OF INVESTMENTS\s+\(in thousands\)\s+(?:\(unaudited\)\s+)?As of\s+\w+\s+\d{1,2},?\s+\d{4}")
    if qtr == '2020-12-31':
        return re.compile(r"TRIPLEPOINT VENTURE GROWTH BDC CORP\. AND SUBSIDIARIES\s+CONSOLIDATED SCHEDULE OF INVESTMENTS\s+\(dollars in thousands\)\s+As of\s+\w+\s+\d{1,2},?\s+\d{4}")
    if qtr == "2021-12-31":
        return re.compile(r"TRIPLEPOINT VENTURE GROWTH BDC CORP\. AND SUBSIDIARIES\s+CONSOLIDATED SCHEDULE OF INVESTMENTS\s+\(dollars in thousands\)\s+As of\s+\w+\s+\d{1,2},?\s+\d{4}")
    if qtr == '2022-12-31':
        return re.compile(r"TRIPLEPOINT VENTURE GROWTH BDC CORP\. AND SUBSIDIARIES\s+CONSOLIDATED SCHEDULE OF INVESTMENTS\s+\(dollars in thousands\)\s+(?:\(unaudited\)\s+)?As of\s+\w+\s+\d{1,2},?\s+\d{4}")
    if qtr in ['2023-06-30','2023-09-30','2023-12-31']:
        return re.compile(r"TRIPLEPOINT VENTURE GROWTH BDC CORP\. AND SUBSIDIARIES\s+CONSOLIDATED SCHEDULE OF INVESTMENTS\s+\(dollars in thousands\)\s+(\(unaudited\)\s+)?As of\s+\w+\s+\d{1,2},\s+\d{4}")
    return re.compile(r"^TRIPLEPOINT VENTURE GROWTH BDC CORP\. AND SUBSIDIARIES\s+CONSOLIDATED SCHEDULE OF INVESTMENTS\s+\((unaudited|dollars in thousands|in thousands)\)\s+\((unaudited|dollars in thousands|in thousands)\)?\s+As of (January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}$|TRIPLEPOINT\s+VENTURE\s+GROWTH\s+BDC\s+CORP\.\s+AND\s+SUBSIDIARIES\s+CONSOLIDATED\s+SCHEDULE\s+OF\s+INVESTMENTS\s+\(dollars\s+in\s+thousands\)\s+\(unaudited\)\s+As\s+of\s+March\s+31,\s+2023|TRIPLEPOINT\s+VENTURE\s+GROWTH\s+BDC\s+CORP\.\s+AND\s+SUBSIDIARIES\s+CONSOLIDATED\s+SCHEDULE\s+OF\s+INVESTMENTS\s+\(unaudited\)\s+\(dollars\s+in\s+thousands\)\s+As\s+of\s+March\s+31,\s+2024")
    
def _clean(
    file_path:str,
    regex_pattern:str=r"^TRIPLEPOINT VENTURE GROWTH BDC CORP\. AND SUBSIDIARIES\s+CONSOLIDATED SCHEDULE OF INVESTMENTS\s+\((unaudited|dollars in thousands|in thousands)\)\s+\((unaudited|dollars in thousands|in thousands)\)?\s+As of (January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}$|TRIPLEPOINT\s+VENTURE\s+GROWTH\s+BDC\s+CORP\.\s+AND\s+SUBSIDIARIES\s+CONSOLIDATED\s+SCHEDULE\s+OF\s+INVESTMENTS\s+\(dollars\s+in\s+thousands\)\s+\(unaudited\)\s+As\s+of\s+March\s+31,\s+2023|TRIPLEPOINT\s+VENTURE\s+GROWTH\s+BDC\s+CORP\.\s+AND\s+SUBSIDIARIES\s+CONSOLIDATED\s+SCHEDULE\s+OF\s+INVESTMENTS\s+\(unaudited\)\s+\(dollars\s+in\s+thousands\)\s+As\s+of\s+March\s+31,\s+2024"
)->pd.DataFrame:
    df = pd.read_csv(file_path,index_col=0,na_values=[' ', ''])
        
    df.replace(['\u200b',None, r'^\s*$'],np.nan,regex=True,inplace=True) #':','$','%'
    df.dropna(axis=0,how='all',inplace=True)
    
    df = df[~df.apply(lambda row:row.astype(str).str.match(regex_pattern).all(),axis=1)]
    important_fields,idx = get_key_fields(df)
    df.columns = important_fields
    df = merge_duplicate_columns(df).reset_index(drop=True)
    
    duplicate_idx = df.apply(lambda row:row[pd.to_numeric(row,errors='coerce').isna()].duplicated().sum() > 1 ,axis=1)
    clean_rows = df.loc[duplicate_idx].apply(remove_row_duplicates, axis=1).reset_index(drop=True)
    j = 0
    for i,flag in enumerate(duplicate_idx):
        if not flag:
            continue
        df.iloc[i,:] = clean_rows.loc[j,:]
        j += 1
    df.replace([r'^\s*$'],np.nan,regex=True,inplace=True) #':','$','%'
    df.dropna(axis=1,how='all',inplace=True)
    
    columns = (~df.isna()).sum(axis=0) <= 3 
    df.drop(columns=df.columns[columns],inplace=True)
    return df.reset_index(drop=True)

def md_parse(
    xml_file:str
):
    # Load HTML content
    with open(xml_file, 'r') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    # If the content is inside <pre>, get the text within it
    pre_text = soup.find('pre').get_text()
    # logger.info(pre_text)
    start = pre_text.split('Portfolio Company')[-1]
    if start == pre_text:
        start = pre_text.split('PORTFOLIO COMPANY')[-1]

    start = 'Portfolio Company' + start[:len(start)]
    end = start.split('NET ASSETS')[0]
    # logger.info(end)
    lines = end.split('\n')
    data = []
    num_cols = len(lines[0].split())
    for line in lines:
        line = line.replace('$','')
        row = [np.nan]*num_cols
        values = re.split(r'\s{4,}', line)
        row[:len(values)] = values
        data.append(row)   

    df = pd.DataFrame(data,columns=data[0]).dropna(axis=1,how='all')
    important_fields,idx = get_key_fields(df)
    df.columns = important_fields
    return df

def exceptions()->tuple:
    return (
        '2015-03-31/Schedule_of_Investments_4.csv',
        '2014-12-31/Schedule_of_Investments_5.csv'
    )

cik = 1580345

def main()->None:
    qtrs = os.listdir('.')
    for qtr in qtrs:
        if '.csv' in qtr or not os.path.exists(os.path.join(qtr,f'Schedule_of_Investments_0.csv')):
            continue
        # qtr = '2015-03-31'
        logger.info(qtr)

        index_list_sum = i = 0
        soi_files = sorted([
            os.path.join(qtr,file) 
            for file in os.listdir(os.path.join(qtr))
            if '.csv' in file
        ],key=lambda f: int(f.split('_')[-1].split('.')[0]))
        df = _clean(soi_files[i],remove_regex(qtr))
        index_list = df.apply(
            lambda row:row.astype(str).str.contains(stopping_criterion(qtr), case=False, na=False).any(),
            axis=1
        )
        index_list_sum = index_list.sum()
        dfs = [df]     
        i += 1
        cols = df.columns.tolist()
        while index_list_sum == 0:
            logger.info(f"WTF {soi_files[i]}")
            if soi_files[i] in exceptions():
                i += 1
                continue
            logger.info(soi_files[i])
            df = _clean(soi_files[i],remove_regex(qtr))
            if set(list(range(10))) >= set(df.columns.tolist()):
                df.columns = cols
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

        date_final['qtr'] = qtr.split('/')[-1]
        if not os.path.exists(os.path.join(qtr,'output')):
            os.makedirs(os.path.join(qtr,'output'))
        columns_to_drop = date_final.notna().sum() <= 2
        date_final.drop(columns=columns_to_drop[columns_to_drop].index)
        date_final.to_csv(os.path.join(qtr,'output',f'{qtr}.csv'),index=False)
        # break
    
    # Use glob to find files
    files = sorted(glob.glob(f'*/output/*.csv'), key=extract_date)
    single_truth = pd.concat([
        pd.read_csv(df) for df in files
    ],axis=0,ignore_index=True)
    single_truth.drop(columns=single_truth.columns[['Unnamed' in col for col in single_truth.columns]],inplace=True)
    single_truth.to_csv(f'{cik}_soi_table.csv',index=False)
    
    
if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    cik = os.path.basename(__file__).split('.py')[0]
    if os.path.exists(f"../logs/{cik}.log"):
        os.remove(f"../logs/{cik}.log")
    logger = init_logger(cik)
    logger.info(cik)
    
    main()