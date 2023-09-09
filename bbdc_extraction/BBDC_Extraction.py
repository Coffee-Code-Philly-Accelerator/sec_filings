import requests
from datetime import datetime
import re
import unicodedata

import logging
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import regex as re

import string
from io import StringIO

logger = logging.getLogger(__name__)
writer = pd.ExcelWriter('BBDC_Final_Test.xlsx')


def parse_and_trim(content, type):
    if type == 'HTML':
        soup = BeautifulSoup(content, 'html.parser')
    else:
        soup = BeautifulSoup(content, 'html.parser')

    for tag in soup.recursiveChildGenerator():
        try:
            tag.attrs = None
        except AttributeError:
            # 'NavigableString' object has no attribute 'attrs'
            pass

    for linebreak in soup.find_all('br'):
        linebreak.extract()
    return soup

def download_file(url):
    headers = {'User-Agent': "jaison.basket@email.com"}
    return requests.get(url,headers=headers)


def get_content(url):
    return download_file(url).content

def formatting_2(file_name, qtr_date):
    pd_csv = pd.read_csv(file_name)
    pd_csv = pd_csv.iloc[:, 1:]
    # Resetting index
    # Find out where portfolio company is
    for row in range(5):
        # print(pd_csv.iloc[row])
        if 'Company' in str(pd_csv.iloc[row]):
            # print("Company Found")
            portfolio_index = row
            break## Changes

    # Checking if all the column are in the row with portfolio and adding them if they are not.
    for j in range(pd_csv.shape[1]):
        if (pd_csv.iat[portfolio_index - 1, j] == 'Principal'):
            pd_csv.iat[portfolio_index, j] = 'Principal'
        elif (pd_csv.iat[portfolio_index - 1, j] == 'Cost'):
            pd_csv.iat[portfolio_index, j] = 'Cost'
        elif (pd_csv.iat[portfolio_index - 1, j] == 'Fair'):
            pd_csv.iat[portfolio_index, j] = 'Fair Value'
        elif ('Principal' in str(pd_csv.iat[portfolio_index - 1, j])):
            pd_csv.iat[portfolio_index, j] = 'Principal'

    # print("Portfolio", portfolio_index)
    # Removing rows before portfolio index
    if (pd_csv.iloc[portfolio_index].str.contains('Company').any() == True):
        # print("Inside")
        pd_csv = pd_csv.iloc[portfolio_index:]

    if (pd_csv['0'].isnull().sum() / len(pd_csv) == 1):
        pd_csv = pd_csv.drop(pd_csv.columns[0], axis=1)

    pd_csv = pd_csv.reset_index(drop=True)
    second_header = list()
    # Identifying where the second header is
    for index, row in pd_csv.iterrows():
        # print(row[0],row[2])
        if index < 5:
            continue
        if 'portfolio' in str(row[1]).lower():
            second_header.append(index)
        elif 'industry' in str(row[2]).lower():
            second_header.append(index)
        elif 'portfolio' in str(row[2]).lower():
            second_header.append(index)
    # Dropping the second header
    try:
        print("Second Header", second_header)
        pd_csv.drop(second_header, inplace=True)
    except:
        print("No second header")

    #Shifting rows back when 1st column is empty
    pd_csv.loc[pd_csv['0'].isna()] = pd_csv.loc[pd_csv['0'].isna()].shift(-1, axis=1)  #Changes

    # Identifying levels in the dataframe
    listOfLevels = list()
    for i in range(pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            if (pd.notnull(pd_csv.iat[i, j])):
                # print(pd_csv.iat[i, j])
                if ((('investments' in str(pd_csv.iat[i, j])) or ('Investments' in str(pd_csv.iat[i, j]))) and
                        ('Subtotal' not in pd_csv.iat[i, j]) and ('Total' not in pd_csv.iat[i, j]) and
                        ('inc' not in str(pd_csv.iat[i, j].lower())) and ('Fund' not in pd_csv.iat[i, j]) and
                        ('LLC' not in pd_csv.iat[i, j]) and ('L.P' not in pd_csv.iat[i, j]) and
                        ('CMS' not in pd_csv.iat[i, j]) and ('holdings' not in str(pd_csv.iat[i, j].lower()))):
                    listOfLevels.append(i)

    # Pattern
    pattern = r"\d{1,3}(,\d{3})*\s+(shares|units|cash|warrant(\s*\(\d+\%\))?)|\d+\%\s+cash|\broyalty\s+rights\b|warrant|units|(preferred interest)|points"

    # Find the index of the "Type of Investment" column
    type_col_idx = pd_csv.iloc[:2].apply(
        lambda x: x.astype(str).str.contains("Type of Investment", case=False)).any().argmax()

    # Shift rows to move matched value in the first column to the column under "Type of Investment"
    for i, row in pd_csv.iterrows():
        if i == 0:
            continue  # Skip the first row
        if bool(re.search(pattern, str(row[0]).lower())):
            # Shift the row to move the matched value in the first column to the column under "Type of Investment"
            pd_csv.loc[i, :] = pd_csv.loc[i, :].shift(3)
        if bool(re.search(pattern, str(row[1]).lower())):
        # Shift the row to move the matched value in the second column to the column under "Type of Investment"
            pd_csv.loc[i,:] = pd_csv.loc[i,:].shift(2)


    remove = string.punctuation
    remove = remove.replace("-", "")  # don't remove hyphens
    pattern = r"[{}]".format(remove)  # create the pattern
    pd_csv = pd_csv.reset_index(drop=True)

    # Adding the Levels to an array
    newValues = []
    # newValues.append("Levels")

    for i in range(len(listOfLevels)):
        beginIndex = listOfLevels[i]
        if (i == len(listOfLevels) - 1):
            endIndex = len(pd_csv)
        else:
            endIndex = listOfLevels[i + 1]
        value = pd_csv.iat[beginIndex, 0]
        if (value == 'Portfolio Company'):
            value = 'Levels'
        try:
            value = ''.join([i for i in value if not i.isdigit()])
            value = re.sub("[^\P{P}-]+", "", value)
        except:
            value = pd_csv.iat[beginIndex, 0]
        # print("Begin", beginIndex)
        # print("End", endIndex)
        for x in range(beginIndex, endIndex):
            newValues.append(value)

    # print(newValues)
    if (len(newValues) < (len(pd_csv))):
        newValues.insert(0, 'Levels')
    # Adding the array to the df
    pd_csv.insert(0, 'Levels', newValues)

    # Finding Subtotals
    pd_csv.reset_index()
    subtotal_list = list()
    total_list = list()
    for i in range(pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            if (('Subtotal' in str(pd_csv.iat[i, j]))):
                # print(pd_csv.iloc[i,:])
                subtotal_list.append(i)

    # print("Subtotal", subtotal_list)
    # Dropping the subtotal rows
    pd_csv.drop(pd_csv.index[subtotal_list], inplace=True)

    # Finding Totals
    total_list = list()
    for i in range(pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            if ('Total' in str(pd_csv.iat[i, j])):
                total_list.append(i)

    # print("Total list", total_list)
    pd_csv.drop(pd_csv.index[total_list], inplace=True)
    # Dropping everything after the total rows
    try:
        pd_csv = pd_csv.iloc[:total_list[-1]]
    except:
        print("Total not found")

    for i in range(5):
        for j in range(pd_csv.shape[1]):
            if ('TypeofInvestment' in str(pd_csv.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                type_of_investment_index = j
                break
            elif ('Company' in str(pd_csv.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                company_first_index = j
    # print("company_first_index", company_first_index)

    for i in range(1):
        for j in range(pd_csv.shape[1]):
            if ('Company' in str(pd_csv.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                company_index = j
            if ('Industry' in str(pd_csv.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                industry_index = j
            elif ('Investment' in str(pd_csv.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                investment_index = j
    # print("Industry index", industry_index)
    # print("Investment index", investment_index)
    # print("Company index", company_index)

    # print(company_first_index)
    # Adding company for the middle values where it might be missed out
    for i in range(2, pd_csv.shape[0]):
        for j in range(company_first_index + 2):
            row_alphabet_check = False
            if (pd.isnull(pd_csv.iat[i, j]) or (pd_csv.iat[i, j] == 'nan')):
                # print("Index",i,j)
                if (i > 0):
                    # print(pd_csv.iat[i-1,j])
                    if (pd.isnull(pd_csv.iat[i - 1, j])):
                        1
                    else:
                        if (re.search('[a-zA-Z]', str(pd_csv.iat[i - 1, j]))):
                            # print(pd_csv.iat[i-1,j])
                            pd_csv.iat[i, j] = pd_csv.iat[i - 1, j]

    industry_col = pd_csv[pd_csv.columns[industry_index]]
    investment_col = pd_csv[pd_csv.columns[investment_index]]
    investment_col_next = pd_csv[pd_csv.columns[investment_index + 1]]
    company_col = pd_csv[pd_csv.columns[company_index]]

    investment_numeric_mask = pd.notna(investment_col) & (
                investment_col.str.isnumeric() | investment_col.str.match('^[-]+$'))
    investment_numeric_indexes = investment_col.loc[investment_numeric_mask].index.to_list()
    industry_numeric_mask = pd.notna(industry_col) & (industry_col.str.isnumeric() | industry_col.str.match('^[-]+$') |
                                                 industry_col.str.contains(r'\b\d{1,2}/\d{1,2}/\d{4}\b')   )
    industry_numeric_indexes = industry_col.loc[industry_numeric_mask].index.to_list()

    investment_in_company_list = list(industry_numeric_indexes) + list(investment_numeric_indexes)

    money_market_index = list()
    for comp_index in investment_in_company_list:
        if 'Money Market Funds' in str(pd_csv.loc[comp_index, pd_csv.columns[company_index]]):
            # print(pd_csv.loc[comp_index, pd_csv.columns[company_index]])
            money_market_index.append(comp_index)

    investment_in_company_list = [x for x in investment_in_company_list if x not in money_market_index]

    pd_csv.loc[investment_in_company_list, pd_csv.columns[company_index]:] = pd_csv.loc[investment_in_company_list,
                                                                             pd_csv.columns[company_index]:].shift(2,
                                                                                                                   axis=1)



    for i in range(5):
        for j in range(pd_csv.shape[1]):
            if ('Industry' in str(pd_csv.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                principal_index = j
                break
    principal_index = 1
    # print("Principal Index", principal_index)

    # Identifying index of maturity
    i = 0
    # Extracting the column data after shares/warrants/principal
    try:
        pd_csv_principal = pd_csv.iloc[:, principal_index:]
        pd_csv_principal = pd_csv_principal.replace("%", np.nan)
        pd_csv_principal = pd_csv_principal.replace("\u200b", np.nan)
        pd_csv_principal = pd_csv_principal.replace("\u200b.1", np.nan)

        pd_csv_principal = pd.read_csv(StringIO(re.sub(',+', ',', pd_csv_principal.to_csv())))

    except:
        print("no Maturity")

    pd_csv_principal = pd_csv_principal.iloc[:, 1:]
    # Dropping columns with Nulls
    perc = 95  # Like N %
    min_count = int(((100 - perc) / 100) * pd_csv_principal.shape[0] + 1)
    pd_csv_principal = pd_csv_principal.dropna(axis=1,
                                               thresh=min_count)

    # Changes

    for i in range(1):
        for j in range(pd_csv_principal.shape[1]):
            # print(i,j)
            # print(str(pd_csv_principal.iat[i, j]).translate({ord(c): None for c in string.whitespace}))
            try:
                if ('NetAssets' in str(pd_csv_principal.iat[i, j]).translate(
                        {ord(c): None for c in string.whitespace})):
                    net_assets_index = j
                    print("Net Assets Index", net_assets_index)
            except:
                print("Net Assets not present")
            if ('FairValue' in str(pd_csv_principal.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                fair_index = j
            if ('AcquisitionDate' in str(pd_csv_principal.iat[i, j]).translate(
                    {ord(c): None for c in string.whitespace})):
                acq_date_index = j
            if ('Investment' in str(pd_csv_principal.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                invest_index = j
            if ('Principal' in str(pd_csv_principal.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                prin_value_index = j
            if ('Maturity' in str(pd_csv_principal.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                maturity_index = j

    # print("Principal Index", principal_index)
    # print("Investment Index", invest_index)
    # print("Principal Value Index", prin_value_index)
    try:
        print("Maturity Index", maturity_index)
    except:
        print("Maturity column missing")
    try:
        print("Acquisition Date Index", acq_date_index)
    except:
        print("No Acq date")


    try:
        pd_csv_principal = pd_csv_principal.iloc[:, :net_assets_index + 1]
    except:
        pd_csv_principal = pd_csv_principal.iloc[:, :fair_index + 1]

    for i in range(int(pd_csv_principal.shape[0])):
        for j in range(pd_csv_principal.shape[1]):
            try:
                pd_csv_principal.iat[i, j] = re.sub(r"\(\d+\)", "", pd_csv_principal.iat[i, j])
                if (pd_csv_principal.iat[i, j] == ''):
                    pd_csv_principal.iat[i, j] = np.nan
            except:
                print("Bracket digits not removed")

    # CHANGES

    principal_cols = pd_csv_principal.columns[principal_index]
    principal_value_cols = pd_csv_principal.columns[prin_value_index]

    try:
        acq_date_cols = pd_csv_principal.columns[acq_date_index]
    except:
        acq_date_cols = list()
        print("No Acq Date")
    try:
        maturity_cols = pd_csv_principal.columns[maturity_index]
    except:
        maturity_cols = list()
        print("No Maturity Column")

    if (len(maturity_cols) > 0 and len(acq_date_cols) > 0):
        last_1_nan_list = pd_csv_principal.index[
            (pd_csv_principal.iloc[:, -1].isnull()) & (pd_csv_principal.iloc[:, -2].notnull()) & (
                pd_csv_principal.iloc[:, -3].notnull())].to_list()
        pd_csv_principal.loc[last_1_nan_list, principal_cols:] = pd_csv_principal.loc[last_1_nan_list,
                                                                 principal_cols:].shift(1, axis=1)
    elif (len(maturity_cols) == 0 and len(acq_date_cols) > 0):
        last_1_nan_list_date_in_invest = pd_csv_principal.index[
            (pd_csv_principal.iloc[:, -1].isnull()) & (pd_csv_principal.iloc[:, -2].notnull()) &
            (pd_csv_principal.iloc[:, -3].notnull()) & pd_csv_principal.iloc[:, invest_index].str.contains(
                '/')].to_list()
        pd_csv_principal.loc[last_1_nan_list_date_in_invest, principal_cols:] = pd_csv_principal.loc[
                                                                                last_1_nan_list_date_in_invest,
                                                                                principal_cols:].shift(1, axis=1)

        last_1_nan_list = pd_csv_principal.index[
            (pd_csv_principal.iloc[:, -1].isnull()) & (pd_csv_principal.iloc[:, -2].notnull()) &
            (pd_csv_principal.iloc[:, -3].notnull())].to_list()
        pd_csv_principal.loc[last_1_nan_list, principal_value_cols:] = pd_csv_principal.loc[last_1_nan_list,
                                                                       principal_value_cols:].shift(1, axis=1)
    # Need to shift if last is Nan and acquisition date has date then do above
    # Otherwise need to shift only the Principal value to the right
    else:
        last_1_nan_list = pd_csv_principal.index[
            (pd_csv_principal.iloc[:, -1].isnull()) & (pd_csv_principal.iloc[:, -2].notnull()) & (
                pd_csv_principal.iloc[:, -3].notnull())].to_list()
        pd_csv_principal.loc[last_1_nan_list, principal_cols:] = pd_csv_principal.loc[last_1_nan_list,
                                                                 principal_cols:].shift(1, axis=1)

    pd_csv = pd_csv.iloc[:, :principal_index].reset_index(drop=True)

    # Concatting everything together
    pd_csv = pd.concat([pd_csv, pd_csv_principal], axis=1)

    perc = 100  # Like N %
    min_count = int(((100 - perc) / 100) * pd_csv.shape[0] + 1)
    # Dropping columns which have 100% NaN
    pd_csv = pd_csv.dropna(axis=1,
                           thresh=min_count)

    new_header = pd_csv.iloc[0]  # grab the first row for the header
    # new_header = new_header.str.replace("Shares/Warrants/Principal","Shares Warrants Principal")
    pd_csv = pd_csv[1:]  # take the data less the header row
    pd_csv.columns = new_header.str.replace(r"\(\d+\)", "")  # set the header row as the df header

    pd_csv = pd_csv.replace("~", ",", regex=True)
    # pd_csv = pd_csv.iloc[:, 1:]
    pd_csv.columns.values[0] = "Levels"  # Dropping columns which NaN as column names
    pd_csv = pd_csv.loc[:, pd_csv.columns.notna()]
    maturity_cols = [col for col in pd_csv.columns if 'Maturity' in col]
    if (len(maturity_cols) > 0):
        pd_csv = pd_csv.dropna(thresh=len(pd_csv.columns) - 4)
    else:
        pd_csv = pd_csv.dropna(thresh=len(pd_csv.columns) - 3)

    perc = 95  # Like N %
    min_count = int(((100 - perc) / 100) * pd_csv.shape[0] + 1)
    pd_csv = pd_csv.dropna(axis=1, thresh=min_count)



    try:
        pd_csv.drop('​.1', inplace=True, axis=1)
    except:
        print(".1 column not present")

    # Appending to Type of Investment from Company column

    type_invest_append = ['(Revolver)', '(First Out)', '(Last Out)', 'Series A-1', 'Series A', 'Series B-1', 'Series B',
                          'Series C', 'Series A-1',
                          'Series B-1', 'Class A', 'Class B', 'Class C', 'Class D', 'Membership Unit Warrants',
                          'Series M-1']
    try:
        comp_cols = [col for col in pd_csv.columns if 'Investment ,' in col]
        type_inv_cols = [col for col in pd_csv.columns if 'Type of Investment' in col]

        invest_append_list = list()
        for i in type_invest_append:
            # print(i)
            if (pd_csv[comp_cols[0]].str.contains(i).any() == True):
                invest_append_list = pd_csv.index[pd_csv[comp_cols[0]].str.contains(i)].to_list()
                for j in invest_append_list:
                    if (i in str(pd_csv.loc[int(j), comp_cols[0]])):
                        pd_csv.loc[int(j), comp_cols[0]] = pd_csv.loc[int(j), comp_cols[0]].replace(i, "")
                        pd_csv.loc[int(j), type_inv_cols[0]] = type_inv_cols[0] + " " + i
    except:
        print("Company Name column different")

    # Replaceing ( with minus
    for i in pd_csv.columns[-3:]:
        pd_csv[i] = pd_csv[i].replace('\\(', '-', regex=True)

    # Changes

    invest_cols = [col for col in pd_csv.columns if 'Investment' in col]
    industry_cols = [col for col in pd_csv.columns if 'Industry' in col]
    acq_date_cols = [col for col in pd_csv.columns if 'Acquisition' in col]
    principal_cols = [col for col in pd_csv.columns if 'Principal' in col]

    last_2_as_nan_list = pd_csv.index[
        (pd_csv.iloc[:, -1].isnull()) & (pd_csv.iloc[:, -2].isnull()) & (pd_csv.iloc[:, -3].notnull())].to_list()
    if (len(maturity_cols) > 0 and len(acq_date_cols) > 0):
        pd_csv.loc[last_2_as_nan_list, maturity_cols[0]:] = pd_csv.loc[last_2_as_nan_list, maturity_cols[0]:].shift(2,
                                                                                                                    axis=1)
    elif (len(maturity_cols) == 0 and len(acq_date_cols) > 0):
        pd_csv.loc[last_2_as_nan_list, industry_cols[0]:] = pd_csv.loc[last_2_as_nan_list, industry_cols[0]:].shift(1,
                                                                                                                    axis=1)
        pd_csv.loc[last_2_as_nan_list, principal_cols[0]:] = pd_csv.loc[last_2_as_nan_list, principal_cols[0]:].shift(1,
                                                                                                                      axis=1)
    else:
        pd_csv.loc[last_2_as_nan_list, invest_cols[0]:] = pd_csv.loc[last_2_as_nan_list, invest_cols[0]:].shift(2,
                                                                                                                axis=1)

    # Changes
    last_3_nan_list = pd_csv.index[
        (pd_csv.iloc[:, -1].isnull()) & (pd_csv.iloc[:, -2].isnull()) & (pd_csv.iloc[:, -3].isnull())].to_list()
    if (len(maturity_cols) > 0):
        industry_cols = [col for col in pd_csv.columns if 'Industry' in col]
        #floor_cols = [col for col in pd_csv.columns if 'Floor' in col]
        pd_csv.loc[last_3_nan_list, industry_cols[0]:] = pd_csv.loc[last_3_nan_list, industry_cols[0]:].shift(1, axis=1)
        #pd_csv.loc[last_3_nan_list, floor_cols[0]:] = pd_csv.loc[last_3_nan_list, floor_cols[0]:].shift(2, axis=1)
    else:
        pd_csv.drop(last_3_nan_list, inplace=True)

    industry_cols = [col for col in pd_csv.columns if 'Industry' in col]
    investements_nan = pd_csv.index[pd_csv[invest_cols[0]].isnull()].to_list()
    pd_csv.loc[investements_nan, [invest_cols[0], industry_cols[0]]] = pd_csv.loc[
        investements_nan, [industry_cols[0], invest_cols[0]]].values

    last_nan = list()
    pd_csv = pd_csv.reset_index(drop=True)

    try:
        for i in range(int(pd_csv.shape[0])):
            if (pd_csv.iat[i, pd_csv.shape[1] - 1] == ' ' or pd_csv.iat[i, pd_csv.shape[1] - 1] == '  '):
                last_nan.append(i)
                pd_csv.iat[i, pd_csv.shape[1] - 1] = np.nan
        # pd_csv.loc[last_nan, maturity_cols[0]:] = pd_csv.loc[last_nan, maturity_cols[0]:].shift(1, axis=1)
    except:
        print("Error while switching for last column")

    principal_cols = [col for col in pd_csv.columns if 'Principal' in col]

    pd_csv[principal_cols[0]] = pd_csv[principal_cols[0]].fillna("")

    # Finding out if the column contains string values, as Principal should only contain numbers.
    alpha_in_principal = pd_csv.index[pd_csv[principal_cols[0]].str.contains(r'[a-zA-Z]')].to_list()

    # Shifting the values back by one column
    pd_csv.loc[alpha_in_principal, industry_cols[0]:principal_cols[0]] = pd_csv.loc[alpha_in_principal,
                                                                         industry_cols[0]:principal_cols[0]].shift(-1,
                                                                                                                   axis=1)
    # print(type_of_investment_index)
    # Adding company for the middle values where it might be missed out
    for i in range(0, pd_csv.shape[0]):
        for j in range(investment_index + 1):
            row_alphabet_check = False
            if (pd.isnull(pd_csv.iat[i, j]) or (pd_csv.iat[i, j] == 'nan')):
                # print("Index",i,j)
                if (i > 0):
                    # print(pd_csv.iat[i-1,j])
                    if (pd.isnull(pd_csv.iat[i - 1, j])):
                        1
                    else:
                        if (re.search('[a-zA-Z]', str(pd_csv.iat[i - 1, j]))):
                            for k in range(pd_csv.shape[1]):
                                if (re.search('[a-zA-Z]', str(pd_csv.iat[i, k])) and pd.notnull(pd_csv.iat[i, k])):
                                    # print("Alpha Check",pd_csv.iat[i,k])
                                    row_alphabet_check = True
                                    break
                            if (row_alphabet_check):
                                # print(pd_csv.iat[i-1,j])
                                pd_csv.iat[i, j] = pd_csv.iat[i - 1, j]

    # Finding the Type of Investment column
    invest_cols = [col for col in pd_csv.columns if 'Investment' in col]
    # Removing rows which contain only numbers in the Type of Investment column
    pd_csv = pd_csv[pd.to_numeric(pd_csv[invest_cols[0]], errors="coerce").isnull()]

    # Finding out if the column Investment contains values like Level 3
    level_in_investments = pd_csv.index[pd_csv[invest_cols[0]].str.contains('Level')].to_list()
    # Dropping it if its there
    pd_csv.drop(level_in_investments, inplace=True)
    #Dropping rows that have only - in Type of Investment
    pd_csv = pd_csv[pd_csv[invest_cols[0]] != '-']
    pd_csv = pd_csv.fillna("")

    # print("Formated Data Frame")
    # print(pd_csv)
    # print("Length",len(pd_csv))
    # pd_csv.to_csv("Formated_" + qtr_date + ".csv")
    pd_csv.to_excel(writer, index=None, header=True, sheet_name=qtr_date)
    #writer.save()


def scrape_data(date):
    #Extracting info from the excel
    outside = False
    inside = False
    data = pd.ExcelFile("bbdc_fillings.xlsx")
    df = data.parse('BBDC')
    url = df['html_link'].str.strip()
    date_filled = df['date_filed']
    #date_filled.head()
    i = 0
    # print("Length of Df",len(df))
    while(i<len(df)):
        #Extracting the content from the URL
        edgar_resp = get_content(url[i]) # TONY uncommented
        edgar_soup = parse_and_trim(edgar_resp, "HTML") # TONY uncommented
        try:
            date_format = datetime.strptime(str(date_filled[i]), "%Y-%m-%d %H:%M:%S")
        except:
            print("Date Error")
        # print("Date of the doc", str(date_format))
        # print("Year", (date_format.strftime("%Y")))
        # print("Month", int(date_format.strftime("%m")))
        if(int(date_format.strftime("%Y"))>=date):
            #print("Scraping Started")
            if(int(date_format.strftime("%m"))<4):
                qtr_date = str("December 31, "+str(int(date_format.strftime("%Y"))-1))
            elif(int(date_format.strftime("%m"))<=5 and int(date_format.strftime("%m"))>3):
                qtr_date = str("March 31, "+str(int(date_format.strftime("%Y"))))
            elif(int(date_format.strftime("%m"))<=8 and int(date_format.strftime("%m"))>6):
                qtr_date = str("June 30, " + str(int(date_format.strftime("%Y"))))
            elif (int(date_format.strftime("%m")) <= 11 and int(date_format.strftime("%m")) > 9):
                qtr_date = str("September 30, " + str(int(date_format.strftime("%Y"))))
            # print("QTR DATE",qtr_date)
            
            # TONY uncommented below
            try:
                formatting_2("output-" + str(i) + "-pandas.csv", qtr_date)
            except:
                print("DID NOT WORK FOR ",str(date_format))
            last_year = (int(date_format.strftime("%Y")) - 2)
            outside_string_check = "December 31, " + str(last_year)
            # print("OUTSIDE STRING CHECK", outside_string_check)
            # print("OUTSIDE EXTRACTION")
            outside = extract_tables_outside(edgar_soup, qtr_date, i, outside_string_check)
            try:
                outside = extract_tables_outside(edgar_soup, qtr_date, i, outside_string_check)
                print("Outside Extraction Done")
            except:
                print("Outside Extract failed")
                outside = 0
            # print("")
            # print("INSIDE EXTRACTION")
            #inside = extract_tables_inside(edgar_soup, qtr_date, i, outside_string_check)
            # print("INSIDE", inside)
            # print("OUTSIDE", outside)
            if (inside == False and outside == 0):
                1
                # print("Group 3 Extraction")
                # extract_tables_3(edgar_soup, qtr_date, i)
                # group3_formatting("output-" + str(i) + "-pandas.csv", qtr_date)
            else:
                formatting_2("output-" + str(i) + "-pandas.csv", qtr_date)
                try:
                    if(outside==2):
                        1
                        #group3_formatting("output-" + str(i) + "-pandas.csv", qtr_date)
                    else:
                        formatting_2("output-" + str(i) + "-pandas.csv", qtr_date)
                except:
                    print("Error:::::::Table not found:::::::::::::")
            formatting_2("output-" + str(i) + "-pandas.csv", qtr_date)

        i = i+1

def extract_tables_outside(soup_content, qtr_date,ind,outside_string_check):
    master_table = None
    return_value = 0
    continous_table_check = 1
    #### find tables when keyword is outside it
    tag_search_test = '^.*(Consolidated Schedule [Oo]f Investments|CONSOLIDATED SCHEDULES OF INVESTMENTS|CONSOLIDATED SCHEDULE OF INVESTMENTS|hedule of Investments|Consolidated\nSchedule of Investments).*$'
    for tag in soup_content.find_all(text=re.compile(tag_search_test)):
        # print("Tag",tag)
        if('as of' in tag or 'and' in tag):
            continue
        date_str= ''
        tag_index = 0
        nextNode = tag
        table_in_tag = 0
        try:
            while tag_index<7:
                nextNode = nextNode.find_next()
                #print(nextNode.find_parent('td').prettify())
                # print("NextNode",nextNode.text)
                try:
                    if 'td' in nextNode.find_parent('td').prettify():
                        table_in_tag = 1
                        print("FOUND TABLE")
                        date_str = qtr_date.lower()
                        break
                except:
                    print("td not found as parent")
                if('Portfolio' in nextNode.findNext('table').text):
                    # print("FOUND PORTFOLIO IN NEXT TABLE")
                    date_str = qtr_date.lower()
                    break
                if(nextNode.text.strip() == 'outside_string_check'):
                    continous_table_check = 0
                    break
                if('Company' in str(nextNode)):
                    # print("STRING CONTAINS TRUE")
                    #print(tag.next_sibling.next_sibling.text.strip().lower())
                    if(qtr_date.lower() in tag.next_sibling.next_sibling.text.strip().lower()):
                        date_str = tag.next_sibling.next_sibling.text.strip().lower()
                        break
                try:
                    if(qtr_date.lower() in unicodedata.normalize('NFKD', nextNode.text.strip()).lower()):
                        print("Date Found")
                        date_str = unicodedata.normalize('NFKD', nextNode.text.strip())
                        break
                    elif(qtr_date.lower() in unicodedata.normalize('NFKD', nextNode.text.strip().replace('\n',' ')).lower()):
                        print("Date Found")
                        date_str = unicodedata.normalize('NFKD', nextNode.text.strip().replace('\n',' '))
                        break
                    # else:
                except AttributeError:
                    print("Error in Date String search")
                tag_index += 1
        except:
            print("No date_str found")
        # print("Date String",date_str)
        first_iteration = True  # initialize flag variable
        if (qtr_date.lower() in date_str.lower() and "for" not in date_str.lower()):
            ###### Use find_previous if the tag is td or tr.
            while(continous_table_check==1):
                if(table_in_tag == 0):
                    if first_iteration:
                        first_iteration = False
                        html_table = nextNode.find_next('table')
                    else:
                        html_table = nextNode
                else:
                    # print("Inside Table")
                    html_table = nextNode.find_parent('table')
                    nextNode = nextNode.find_parent('table')
                    # print("IN NEXT NODE",nextNode.find_parent('table'))
                    table_in_tag = 0
                    first_iteration = False
                # print("HTML TABLE",html_table.prettify())
                if master_table is None:
                    master_table = pd.read_html(html_table.prettify(), skiprows=0, flavor='bs4')[0]
                    master_table = master_table.applymap(lambda x: unicodedata.normalize('NFKD', x.strip().strip(u'\u200b').replace('—','-')) if type(x) == str else x)
                    master_table = master_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan,
                                                                                              regex=True)
                    master_table = master_table.dropna(how='all', axis=0)
                    try:
                        last_row = master_table.iloc[-1, 0].replace(" ", "")
                        print("Last Row",last_row)
                        if ('TOTALINVESTMENTS' in last_row):
                            continous_table_check = 0
                            print('Total investments founds')
                            break
                    except:
                        print("Float object")

                else:
                    try:
                        new_table = pd.read_html(html_table.prettify(), skiprows=0, flavor='bs4')[0]
                        new_table = new_table.applymap(lambda x: unicodedata.normalize('NFKD', x.strip().strip(u'\u200b').replace('—','-')) if type(x) == str else x)
                        new_table = new_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan,
                                                                                                  regex=True)
                        new_table = new_table.dropna(how='all', axis=0)
                    except:
                        print("Go to next")
                    # try:
                    #print('Total Investments, '+qtr_date)
                    #print(' Total Portfolio Investments, '+qtr_date)
                    # print("Last Row",str(new_table.iloc[-1,0]).replace(" ", ""))
                    last_row = str(new_table.iloc[-1,0]).replace(" ", "")
                    if('TOTALINVESTMENTS' in last_row):
                        # print("Total Investment found")
                        master_table = master_table.append(
                            new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
                            ignore_index=True)
                        continous_table_check = 0
                        print('Total investments founds')
                        break
                    elif('TOTALDERIVATIVES' in last_row):
                        master_table = master_table.append(
                            new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
                            ignore_index=True)
                        continous_table_check = 0
                        # print('TOTAL DERIVATIVES founds')
                        break
                    elif ('TotalInvestments' in last_row):
                        master_table = master_table.append(
                            new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
                            ignore_index=True)
                        continous_table_check = 0
                        # print('TOTAL DERIVATIVES founds')
                        break
                    elif ('TotalPortfolioInvestments' in last_row):
                        master_table = master_table.append(
                            new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
                            ignore_index=True)
                        continous_table_check = 0
                        # print('TOTAL DERIVATIVES founds')
                        break
                    # print("Full Last Row of new table",new_table.iloc[-1])
                    for i in range(len(new_table.iloc[-1])):
                        if ('Total Portfolio Investments' in str(new_table.iloc[-1, i])):
                            master_table = master_table.append(
                                new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
                                ignore_index=True)
                            continous_table_check = 0
                            # print('Total Portfolio Investments')
                            break
                    try:
                        # for col in new_table.columns:
                        #     if (new_table[col].astype(str).eq("(1)").any() == False):
                        #         print("End not found")
                        #     else:
                        #         print("NEW TABLE COLUMN",new_table[col])
                        #         print("End of continous table")
                        #         continous_table_check = 0
                        #         break
                        if (continous_table_check == 1):
                            master_table = master_table.append(
                                new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
                                ignore_index=True)
                    except:
                        print("no new table")
                #nextNode = nextNode.find_next('table')
                #print("Last Next Node",nextNode)
                nextNode = nextNode.findNext('table')
        if(continous_table_check == 0):
            # print("BREAKING")
            break
    if master_table is not None:
        # print("INSIDE NOT NONE")
        master_table = master_table.applymap(lambda x: x.strip().strip(u'\u200b') if type(x) == str else x)
        master_table = master_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan, regex=True).replace(r'^\s*\)\s*$', np.nan, regex=True)
        master_table = master_table.replace("~", ",", regex=True)
        ## Code to understand if keyword is inside table or outside
        master_table.to_csv("output-" + str(ind) + "-outside-pandas.csv") # TONY CHANGED HERE
        count = 0
        for col in master_table.columns:
            try:
                count = count + 1
                if (master_table[col].astype(str).str.contains(outside_string_check).any() == True):
                    1
                    # print("Outside String found")
                else:
                    master_table.to_csv("output-" + str(ind) + "-pandas.csv") # TONY CHANGED HERE
                    return_value = 1
            except:
                print("not running full cols")
            if count > 2:
                break
        # print("RETURN VALUE",return_value)
        if(return_value == 1):
            print("MASTER TABLE",master_table)
            master_table.to_csv("output-" + str(ind) + "-pandas.csv")
            return 1
    else:
        # print("Master table none")
        return 0
    #return html_table

def extract_tables_inside(soup_content, qtr_date,ind,outside_string_check):
    master_table = None
    #### Find all tables when the keyword is inside the tablee
    for tag in soup_content.find_all("table"):
        for tag_child in tag.find_all(text=re.compile('^.*(Consolidated Schedule [Oo]f Investments|SCHEDULE OF INVESTMENTS|Schedule of Investments).*$')):
            #print("Tag Child",tag_child)
            for tag_children in tag.find_all(text=re.compile("As of "+qtr_date)):
                #print("Tag Children",tag_children)
                if(qtr_date in tag_children):
                    master_table = pd.read_html(tag.prettify(), skiprows=0, flavor='bs4')[0]
                    master_table = master_table.applymap(
                        lambda x: unicodedata.normalize('NFKD', x.strip().strip(u'\u200b').replace('—', '-')) if type(
                            x) == str else x)
                    master_table = master_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan,
                                                                                              regex=True)
                    master_table = master_table.dropna(how='all', axis=0)
    if master_table is not None:
        master_table = master_table.applymap(lambda x: x.strip().strip(u'\u200b') if type(x) == str else x)
        master_table = master_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan, regex=True).replace(r'^\s*\)\s*$', np.nan, regex=True)
        master_table = master_table.replace("~", ",", regex=True)
        ## Code to understand if keyword is inside table or outside
        master_table.to_csv("output-" + str(ind) + "inside-pandas.csv") # TONY CHANGED HERE
        for col in master_table.columns:
            if (master_table[col].astype(str).str.contains(outside_string_check).any() == True ):
                print("Inside Extraction Failed")
                return False
            else:
                master_table.to_csv("output-"+str(ind)+"-pandas.csv")
                return True
    else:
        #print("INSIDE EXTRACTION FAILED")
        return False
if __name__ == "__main__":
    scrape_data(2012)
    writer.save()

