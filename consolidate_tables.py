import os
import re
import logging
import warnings
import glob
import pandas as pd
import numpy as np
import platform
from collections import Counter
from functools import reduce
from fuzzywuzzy import process

from utils import arguements,init_logger,ROOT_PATH,extract_date,concat,remove_row_duplicates


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
    )
    
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


def stopping_criterion(
    search_string:str='total investments'
)->str:
    # Regular expression to ignore whitespace and case
    regex_pattern = search_string.replace(' ', r'\s*')
    return '{}|{}'.format(regex_pattern,'Invesmtents')

def extract_subheaders(
    df:pd.DataFrame,
)->pd.DataFrame:
    # include = df.apply(
    # lambda row: row.astype(str).str.contains('|'.join(common_subheaders()), case=False, na=False).any(),
    #     axis=1) # 
    include = df.apply(
        lambda row: re.search('|'.join(common_subheaders()), str("".join(row.astype(str))), re.IGNORECASE) is not None,#row.astype(str).str.contains('|'.join(common_subheaders()), case=False, na=False).any(),
        axis=1
    )  
    exclude = ~df.apply(
        lambda row: row.astype(str).str.contains('total', case=False, na=False).any(),
        axis=1
    )
    idx = df[include & exclude].index.tolist()
    df['subheaders'] = 'no_subheader'
    if not idx:
        return df

    prev_header = subheader = None
    df.loc[idx[-1]:,'subheaders'] = df.iloc[idx[-1],1] if isinstance(df.iloc[idx[-1],0],float)  else df.iloc[idx[-1],0]
    for j,i in enumerate(idx[:-1]):
        prev_header = subheader
        subheader = df.iloc[i,1] if isinstance(df.iloc[i,0],float)  else df.iloc[i,0]
        df.loc[idx[j]:idx[j+1],'subheaders'] = subheader if subheader != '' else prev_header
    df.drop(idx,axis=0,inplace=True,errors='ignore') # drop subheader row
    return df

def _clean_qtr(
    file_path:str
)->pd.DataFrame:
    df = pd.read_csv(file_path,index_col=0)
    df.replace(['\u200b','',')',':','$','%',0],np.nan,inplace=True) #':','$','%'
    df.dropna(axis=1,how='all',inplace=True)
    df.dropna(axis=0,how='all',inplace=True)
    
    cols = df.iloc[0].str.replace('[^a-zA-Z]', '', regex=True)
    if ((cols == '') + (cols == 'nan') + (cols == np.nan) + (cols == 'NaN')).sum() > int(file_path.split('\\')[-3] == '1396440'):
        df = df.apply(remove_row_duplicates, axis=1)
    else:
        df.columns = cols
    df = merge_duplicate_columns(df)
    df.replace(['\u200b','',')',':','$','%',0],np.nan,inplace=True) #':','$','%'
    if ((cols == '') + (cols == 'nan') + (cols == np.nan) + (cols == 'NaN')).sum() > int(file_path.split('\\')[-3] == '1396440'):
        df.dropna(axis=1,how='all',inplace=True)
        df.dropna(axis=0,how='all',inplace=True)
    # df = df.iloc[1: , :]
    df.drop(columns=df.columns[pd.isna(df.columns)].tolist() + [col for col in df.columns if col == ''],axis=1,inplace=True)
    return df



def clean(
    file:str,
)->list:
    dirs = file.split('/') if platform.system() == "Linux" else file.split('\\')
    if  len(dirs) < 3 or '.csv' not in dirs[-1]:
        return
    df_cur = pd.read_csv(file,encoding='utf-8')
    df_cur = df_cur.T.drop_duplicates().T
    if df_cur.shape[1] < 4:
        return
    if df_cur.empty:
        return
    
    df_cur.reset_index(drop=True,inplace=True)
    
    important_fields,idx = get_key_fields(df_cur)
    if len(set(important_fields) - {''}) < 4:
        df_cur.replace('\u200b', np.nan, regex=True,inplace=True)
        df_cur.replace(r'\$|€|£',np.nan,regex=True,inplace=True)
        columns_to_drop = df_cur.notna().sum() == 1
        return df_cur.iloc[:,1:].drop(columns=columns_to_drop[columns_to_drop].index)
    
    df_cur.columns = important_fields
    df_cur = merge_duplicate_columns(df_cur)
    cols_to_drop = [
        col for col in df_cur.columns.tolist() 
        if col == '' or col == 'nan'
    ] 

    df_cur.drop(columns=cols_to_drop, errors='ignore',inplace=True) # drop irrelevant columns
    df_cur.dropna(axis=0,how='all',inplace=True)
    return df_cur


def strip_string(
    columns_names:list,
    standardize:bool=False
)->tuple:
    columns = tuple(map(lambda col:re.sub(r'[^a-z]', '', str(col).lower()),columns_names))
    if standardize:
        standard_fields = standard_field_names()
        return tuple(
            get_standard_name(col,standard_fields) for col in columns
        )
    return columns

def get_key_fields(
    df_cur:pd.DataFrame
)->tuple:
    important_fields = standard_field_names()
    for idx,row in enumerate(df_cur.iterrows()):
        found = any(any(
            key in str(field).lower() 
            for key in important_fields)
                    for field in row[-1].dropna().tolist()
            )
        if found and len(set(row[-1].dropna().tolist())) >= 5:
            fields = strip_string(row[-1].tolist(),standardize=found) ,idx
            return fields
    return strip_string(df_cur.iloc[0].tolist(),standardize=found),0

 
def get_standard_name(col, choices, score_cutoff=60):
    best_match, score = process.extractOne(col, choices)
    if score > score_cutoff:
        return best_match
    return col


def exceptions(
    date:str,
    cik:str,
    dfs:list,
)->list:
    warning = 'EXCEPTION'
    #TODO how to generalize below if statement?
    if cik == '1501729' and date == '2012-12-31':#1501729\2012-12-31
        dfs[-1].drop(columns=dfs[-1].columns[2],inplace=True)
        logging.debug(f'{warning} - {dfs[-1]}')
    if cik == '1422183' and date == '2012-03-31':
        dfs[-1].drop(dfs[-1].columns[[2,4,7]],axis=1,inplace=True)
        logging.debug(f'{warning} - {dfs[-1]}')
    if cik == '1422183' and (date == '2012-06-30' or date == '2012-12-31'):
        dfs[-2].drop(dfs[-2].columns[[2]],axis=1,inplace=True)
        logging.debug(f'{warning} - {dfs[-1]}')
    return dfs

def process_date(
    date:str,
    cik:str,
)->dict:
    if not os.path.exists(f"{ROOT_PATH}/{cik}/{date}/output"):
        os.mkdir(f"{ROOT_PATH}/{cik}/{date}/output") 
    files = os.listdir(os.path.join(ROOT_PATH,cik,date))
    files = sorted(
        files, 
        key=lambda file: int(file.split('_')[-1].replace(".csv","")) if file.split('_')[-1].replace(".csv","").isdigit() else 999
    )
    df_cur = clean(os.path.join(ROOT_PATH,cik,date,files[0]))
    for i,file in enumerate(files[1:]):
        if df_cur is None or df_cur.empty:
            df_cur = clean(os.path.join(ROOT_PATH,cik,date,file))
            continue
            
        df_cur.to_csv(f"{ROOT_PATH}/{cik}/{date}/output/cleaned_{i}.csv")
        index_list = df_cur.apply(
            lambda row:row.astype(str).str.contains('total investments', case=False, na=False).any(),
            axis=1
        )
        if index_list.sum() > 0:
            break
        df_cur = clean(os.path.join(ROOT_PATH,cik,date,file))
    cleaned = os.listdir(f'{ROOT_PATH}/{cik}/{date}/output')
    
    if not cleaned:
        return
    
    cleaned = sorted(
        cleaned, 
        key=lambda file: int(file.split('_')[-1].replace(".csv","")) if file.split('_')[-1].replace(".csv","").isdigit() else 999
    )
    dfs = [
        pd.read_csv(os.path.join(f"{ROOT_PATH}/{cik}/{date}/output",f"{file}")) 
        for file in cleaned
    ]
    final_columns = dfs[0].columns

    dfs = exceptions(date,cik,dfs)
    date_final = pd.DataFrame(concat(*dfs))

    date_final.columns = final_columns
    if not os.path.exists(f"{ROOT_PATH}/{cik}/{date}/output_final"):
        os.mkdir(f"{ROOT_PATH}/{cik}/{date}/output_final")
    
    date_final.drop(date_final.columns[0],axis=1,inplace=True)
    date_final = extract_subheaders(date_final)
    date_final['date'] = date
    date_final.reset_index(inplace=True,drop=True)
    date_final.to_csv(f"{ROOT_PATH}/{cik}/{date}/output_final/{date}_final.csv")
    
            
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

def join_all_possible(
    cik:str    
)->None:
    infile = f'{ROOT_PATH}/{cik}/*/output_final/*'
    all_csvs = glob.glob(infile,recursive=True)
    dfs = [pd.read_csv(csv) for csv in all_csvs]
    merged_df = pd.concat(dfs)
    merged_df.drop(columns=merged_df.columns[0],inplace=True)
    merged_df.reset_index(inplace=True)
    merged_df.rename({'Index':'original_index'},inplace=True)
    
    # Seperates totals from soi tables
    mask = merged_df[merged_df['portfolio'].str.contains('total investments|liabilities|net assets|total equity/other',case=False,na=False)].index.to_numpy()
    extracted_rows = merged_df.loc[mask]
    extracted_rows.dropna(axis=1,how='all').drop(['subheaders'],axis=1).to_csv(f'{cik}/totals.csv')
    
    logging.debug(f"final table shape - {merged_df.shape}")
    merged_df.dropna(axis=0,thresh=(merged_df.shape[1] - 7),inplace=True) # drop empty 
    merged_df.reset_index(inplace=True,drop=True)
    logging.debug(f"NULL SUBHEADERS - {merged_df.subheaders.isnull().sum()}\n{merged_df.subheaders.apply(lambda x: type(x).__name__).unique()}")

    merged_df.to_csv(f'{ROOT_PATH}/{cik}/{cik}_soi_table.csv')   
    return

def validate_totals(
    soi:pd.DataFrame,
    totals:pd.DataFrame,
    cik:str,
)->bool:
    totals = totals[totals['portfolio'].str.contains('total investments', case=False, na=False)][['date','cost','value']].reset_index()
    totals.cost = totals.cost.replace(r'[^\d\.-]', '', regex=True).apply(pd.to_numeric)
    totals.value = totals['value'].replace(r'[^\d\.-]', '', regex=True).apply(pd.to_numeric)
    
    soi.cost = soi.cost.str.replace(r'[^\d\.-]', '', regex=True).apply(pd.to_numeric)
    soi.value = soi.value.str.replace(r'[^\d\.-]', '', regex=True).apply(pd.to_numeric)
    soi_totals = soi.groupby(['date']).agg({'cost':'sum','value':'sum'}).reset_index()

    for i in range(soi_totals.shape[0]):
        try:
            assert np.allclose(
                soi_totals[['cost','value']].loc[i].to_numpy(), 
                totals[['cost','value']].loc[i].to_numpy(),
                atol=1000
            ),f"Test {totals['date'].loc[i]} - Failed"
            logging.info(f"Test {totals['date'].loc[i]} - Passed")
        except AssertionError as e:
            logging.error(e)
    
    totals.merge(
        soi_totals, 
        on='date', 
        how='inner',
        suffixes=('_published', '_aggregate')
    ).reset_index().drop(['index','level_0'],axis=1).to_csv(f'{cik}/totals_validation.csv',index=False)

def case_fsk(
    cik:int
)->None:
    for date in os.listdir(f'{ROOT_PATH}/{cik}'):
        if '.csv' in date:
            continue
        logging.info(f"DATE - {date}")
        process_date(date,cik)
    join_all_possible(cik)
    # TODO fix unit testing for other BDC
    # validate_totals(pd.read_csv(f'{cik}/soi_table_all_possible_merges.csv'),pd.read_csv(f'{cik}/totals.csv'),cik=cik)
    return

def case_main(
    cik:int
)->None:
    bdc = os.path.join(ROOT_PATH,str(cik))
    qtrs = os.listdir(bdc)
    for qtr in qtrs:
        if '.csv' in qtr:
            continue
        dfs = []
        index_list_sum = i = 0
        while index_list_sum == 0:
            qtr_file = os.path.join(bdc,qtr,f'Schedule_of_Investments_{i}.csv')
            df = _clean_qtr(qtr_file)
            dfs.append(df)
            index_list = df.apply(
                lambda row:row.astype(str).str.contains('total portfolio investments', case=False, na=False).any(),
                axis=1
            )
            index_list_sum = index_list.sum()
            i += 1
        
        date_final = pd.concat(dfs,axis=0,ignore_index=True) if cik == '1396440' else pd.DataFrame(concat(*dfs))        
        date_final = extract_subheaders(date_final)
        date_final['qtr'] = qtr.split('\\')[-1]
        for i in range(3):
            date_final[date_final.columns[i]].fillna(method='ffill',inplace=True)
        if cik == '1396440':
            idx = date_final[['Principal', 'Cost', 'FairValue']].isna().all(axis=1)
            date_final = date_final.loc[~idx,:]
        if not os.path.exists(os.path.join(bdc,qtr,'output')):
            os.makedirs(os.path.join(bdc,qtr,'output'))
        date_final.to_csv(os.path.join(bdc,qtr,'output',f'{qtr}.csv'),index=False)

    # Use glob to find files
    files = sorted(glob.glob(f'{cik}/*/output/*.csv'), key=extract_date)
    single_truth = pd.concat([
        pd.read_csv(df) for df in files
    ],axis=0,ignore_index=True)
    single_truth.columns = single_truth.columns if cik == '1396440' else single_truth.iloc[0,:-2].tolist() + ['subheaders','qtr']

    single_truth.to_csv(os.path.join(str(cik),f'{cik}_soi_table.csv'),index=False)

def main()->None:
    warnings.simplefilter(action='ignore', category=FutureWarning)
    args = arguements()
    cik = args.cik
    if not os.path.exists(f'{ROOT_PATH}/{cik}'):
        os.mkdir(f'{ROOT_PATH}/csv')
    if cik == '1501729' or cik == '1422183':
        case_fsk(cik)
    if cik == '1396440' or cik == '1490349':
        case_main(cik)
    return 

if __name__ == "__main__":
    """
    python .\consolidate_tables.py --cik 1501729 --url-txt urls/1501729.txt --x-path xpaths/1501729.txt
    python .\consolidate_tables.py --cik 1422183 --url-txt urls/1422183.txt --x-path xpaths/1422183.txt
    python .\consolidate_tables.py --cik 1396440 --url-txt urls/1396440.txt --x-path xpaths/1396440.txt
    
    remove files that don't contain keyword
    https://unix.stackexchange.com/questions/150624/remove-all-files-without-a-keyword-in-the-filename 
    https://stackoverflow.com/questions/26616003/shopt-command-not-found-in-bashrc-after-shell-updation
    """
    init_logger()
    main()
