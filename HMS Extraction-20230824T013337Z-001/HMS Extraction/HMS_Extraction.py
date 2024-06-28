import requests
from datetime import datetime
import re
import unicodedata
import pandas as pd
import numpy as np
import imgkit
import json
import logging
from bs4 import BeautifulSoup
import webbrowser, os
import json
import boto3
import io
from io import BytesIO
import sys
from pprint import pprint
from io import StringIO

logger = logging.getLogger(__name__)
writer = pd.ExcelWriter('HMS_Final_Test.xlsx')


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

def formatting(file_name,qtr_date):
    pd_csv = pd.read_csv(file_name)
    pd_csv = pd_csv.iloc[:, 1:]
    # Resetting index
    pd_csv.reset_index()
    # Find out where portfolio company is
    for col in pd_csv.columns:
        pd_csv[col] = pd_csv[col].astype(str).str.replace('  ', ' ')
        portfolio_index_list = pd_csv.index[pd_csv[col].astype(str).str.contains("Portfolio Company")].to_list()
        if (len(portfolio_index_list) > 0):
            break
    portfolio_index = portfolio_index_list[0]
    # Checking if all the column are in the row with portfolio and adding them if they are not.
    if (pd_csv.iloc[portfolio_index].str.contains('Principal').any() == False):
        for j in range(pd_csv.shape[1]):
            if (pd_csv.iat[portfolio_index - 1, j] == 'Principal'):
                pd_csv.iat[portfolio_index, j] = 'Principal'
            elif (pd_csv.iat[portfolio_index - 1, j] == 'Cost'):
                pd_csv.iat[portfolio_index, j] = 'Cost'
            elif (pd_csv.iat[portfolio_index - 1, j] == 'Fair'):
                pd_csv.iat[portfolio_index, j] = 'Fair Value'
    # Removing rows before portfolio index
    if (pd_csv.iloc[portfolio_index].str.contains('Principal').any() == True):
        pd_csv = pd_csv.iloc[portfolio_index:]

    # Identifying levels in the dataframe
    listOfLevels = list()
    for i in range(pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            if (pd.notnull(pd_csv.iat[i, j])):
                # #print(pd_csv.iat[i, j])
                if (('Investments' in str(pd_csv.iat[i, j])) and
                        ('Subtotal' not in pd_csv.iat[i, j]) and
                        ('Total' not in pd_csv.iat[i, j])):
                    if (pd_csv.iloc[i].isnull().sum() == pd_csv.shape[1] - 1):
                        # #print(pd_csv.iloc[i])
                        listOfLevels.append(i)

    # Adding the Levels to an array
    newValues = []
    newValues.append("Levels")
    for i in range(len(listOfLevels)):
        beginIndex = listOfLevels[i]
        if (i == len(listOfLevels) - 1):
            endIndex = len(pd_csv)
        else:
            endIndex = listOfLevels[i + 1]
        value = pd_csv.iat[beginIndex, 0]
        for x in range(beginIndex, endIndex):
            newValues.append(value)
    # Adding the array to the df
    pd_csv.insert(0, 'Levels', newValues)

    # Finding Subtotals
    pd_csv.reset_index()
    subtotal_list = list()
    total_list = list()
    for i in range(pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            if (('Subtotal' in str(pd_csv.iat[i, j]))):
                # #print(pd_csv.iloc[i,:])
                subtotal_list.append(i)

    ##print("Subtotal", subtotal_list)
    # Dropping the subtotal rows
    pd_csv.drop(pd_csv.index[subtotal_list], inplace=True)

    # Finding Totals
    total_list = list()
    for i in range(pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            if ('Total' in str(pd_csv.iat[i, j])):
                total_list.append(i)
    # Dropping everything after the total rows
    try:
        pd_csv = pd_csv.iloc[:total_list[0]]
    except:
        print("Total not found")

    # Adding company for the middle values where it might be missed out
    for i in range(2, pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            row_alphabet_check = False
            if (pd.isnull(pd_csv.iat[i, j]) or (pd_csv.iat[i, j] == 'nan')):
                # #print("Index",i,j)
                if (i > 0):
                    # #print(pd_csv.iat[i-1,j])
                    if (pd.isnull(pd_csv.iat[i - 1, j])):
                        1
                    else:
                        if (re.search('[a-zA-Z]', str(pd_csv.iat[i - 1, j]))):
                            for k in range(pd_csv.shape[1]):
                                if (re.search('[a-zA-Z]', str(pd_csv.iat[i, k])) and pd.notnull(pd_csv.iat[i, k])):
                                    # #print("Alpha Check",pd_csv.iat[i,k])
                                    row_alphabet_check = True
                                    break
                            if (row_alphabet_check):
                                # #print(pd_csv.iat[i-1,j])
                                pd_csv.iat[i, j] = pd_csv.iat[i - 1, j]

    # Identifying index of Principal
    i = 0
    for j in range(pd_csv.shape[1]):
        if ('Principal' in str(pd_csv.iat[i, j])):
            principal_index = j
    # Extracting the column data after principal
    pd_csv_principal = pd_csv.iloc[:, principal_index - 1:]
    # Sweeping everyhing to left in extracted df
    pd_csv_principal = pd.read_csv(StringIO(re.sub(',+', ',', pd_csv_principal.to_csv())))
    pd_csv_principal = pd_csv_principal.iloc[:, 1:]
    # Dropping columns with Nulls
    perc = 95  # Like N %
    min_count = int(((100 - perc) / 100) * pd_csv_principal.shape[0] + 1)
    pd_csv_principal = pd_csv_principal.dropna(axis=1,
                                               thresh=min_count)
    pd_csv_principal.update(
        pd_csv_principal.loc[pd_csv_principal.iloc[:, -1].isna(), :].shift(axis=1).replace(np.nan, 'na')
    )
    pd_csv_principal.replace('na', np.nan, inplace=True)
    pd_csv = pd_csv.iloc[:, :principal_index-1].reset_index()
    # Concatting everything together
    pd_csv = pd.concat([pd_csv, pd_csv_principal], axis=1)

    # Sweeping to left for member units
    pd_csv_sweep_left = pd_csv.iloc[:, :(principal_index + 2)]
    pd_csv_sweep_left = pd.read_csv(StringIO(re.sub(',+', ',', pd_csv_sweep_left.to_csv())))
    pd_csv_sweep_right = pd_csv.iloc[:, principal_index + 2:]
    pd_csv = pd.concat([pd_csv_sweep_left, pd_csv_sweep_right], axis=1)

    perc = 100  # Like N %
    min_count = int(((100 - perc) / 100) * pd_csv.shape[0] + 1)
    # Dropping columns which have 100% NaN
    pd_csv = pd_csv.dropna(axis=1,
                           thresh=min_count)

    new_header = pd_csv.iloc[0]  # grab the first row for the header
    pd_csv = pd_csv[1:]  # take the data less the header row
    pd_csv.columns = new_header.str.replace(r"\(\d+\)", "")  # set the header row as the df header

    pd_csv = pd_csv.iloc[:, 1:]
    pd_csv = pd_csv.replace("~", ",", regex=True)
    pd_csv = pd_csv.iloc[:, 1:]

    # Dropping columns which NaN as column names
    pd_csv = pd_csv.loc[:, pd_csv.columns.notna()]
    pd_csv = pd_csv.dropna(thresh=len(pd_csv.columns) - 1)
    pd_csv = pd_csv.fillna("")
    try:
        for col in pd_csv.columns:
            # #print("Column", col)
            pd_csv[col] = pd_csv[col].astype(str).str.replace(r"\(\d+\)", "", regex=True)
    except:
        print("Bracket digits not removed")
    ##print(pd_csv)
    #pd_csv.to_csv("Formated_" + qtr_date + ".csv")
    pd_csv.to_excel(writer, index=None, header=True, sheet_name=qtr_date)
    #writer.save()

def group3_formatting(file_name,qtr_date):
    pd_csv = pd.read_csv(file_name)
    pd_csv = pd_csv.iloc[:, 1:]
    # Resetting index
    pd_csv.reset_index()
    # Find out where portfolio company is
    for col in pd_csv.columns:
        pd_csv[col] = pd_csv[col].astype(str).str.replace('  ', ' ')
        portfolio_index_list = pd_csv.index[pd_csv[col].astype(str).str.contains("Portfolio Company")].to_list()
        if (len(portfolio_index_list) > 0):
            break
    portfolio_index = portfolio_index_list[0]
    # Checking if all the column are in the row with portfolio and adding them if they are not.
    if (pd_csv.iloc[portfolio_index].str.contains('Principal').any() == False):
        for j in range(pd_csv.shape[1]):
            if (pd_csv.iat[portfolio_index - 1, j] == 'Principal'):
                pd_csv.iat[portfolio_index, j] = 'Principal'
            elif (pd_csv.iat[portfolio_index - 1, j] == 'Cost'):
                pd_csv.iat[portfolio_index, j] = 'Cost'
            elif (pd_csv.iat[portfolio_index - 1, j] == 'Fair'):
                pd_csv.iat[portfolio_index, j] = 'Fair Value'
    # Removing rows before portfolio index
    if (pd_csv.iloc[portfolio_index].str.contains('Principal').any() == True):
        pd_csv = pd_csv.iloc[portfolio_index:]

    # Identifying levels in the dataframe
    listOfLevels = list()
    for i in range(pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            if (pd.notnull(pd_csv.iat[i, j])):
                # #print(pd_csv.iat[i, j])
                if (('Investments' in str(pd_csv.iat[i, j])) and
                        ('Subtotal' not in pd_csv.iat[i, j]) and
                        ('Total' not in pd_csv.iat[i, j])):
                    if (pd_csv.iloc[i].isnull().sum() == pd_csv.shape[1] - 1):
                        # #print(pd_csv.iloc[i])
                        listOfLevels.append(i)
    #print(listOfLevels)

    # Adding the Levels to an array
    newValues = []
    newValues.append("Levels")
    for i in range(len(listOfLevels)):
        beginIndex = listOfLevels[i]
        if (i == len(listOfLevels) - 1):
            endIndex = len(pd_csv)
        else:
            endIndex = listOfLevels[i + 1]
        value = pd_csv.iat[beginIndex, 0]
        for x in range(beginIndex, endIndex):
            newValues.append(value)

    # Adding the array to the df
    pd_csv.insert(0, 'Levels', newValues)
    pd_csv = pd_csv.drop(listOfLevels)

    # Finding Subtotals
    pd_csv.reset_index()
    subtotal_list = list()
    total_list = list()
    for i in range(pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            if (('Subtotal' in str(pd_csv.iat[i, j]))):
                # #print(pd_csv.iloc[i,:])
                subtotal_list.append(i)

    #print("Subtotal", subtotal_list)
    # Dropping the subtotal rows
    pd_csv.drop(pd_csv.index[subtotal_list], inplace=True)

    # Finding Totals
    total_list = list()
    for i in range(pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            if ('Total' in str(pd_csv.iat[i, j])):
                total_list.append(i)
    # Dropping everything after the total rows
    try:
        pd_csv = pd_csv.iloc[:total_list[0]]
    except:
        print("Total not found")

    # Identifying index of Type of Investment
    i = 0
    for j in range(pd_csv.shape[1]):
        if ('Type of Investment' in str(pd_csv.iat[i, j])):
            type_of_investment_index = j
    #print("Type Investment", type_of_investment_index)

    # Adding company for the middle values where it might be missed out
    for i in range(2, pd_csv.shape[0]):
        for j in range(type_of_investment_index):
            row_alphabet_check = False
            if (pd.isnull(pd_csv.iat[i, j]) or (pd_csv.iat[i, j] == 'nan')):
                # #print("Index",i,j)
                if (i > 0):
                    # #print(pd_csv.iat[i-1,j])
                    if (pd.isnull(pd_csv.iat[i - 1, j])):
                        1
                    else:
                        if (re.search('[a-zA-Z]', str(pd_csv.iat[i - 1, j]))):
                            for k in range(pd_csv.shape[1]):
                                if (re.search('[a-zA-Z]', str(pd_csv.iat[i, k])) and pd.notnull(pd_csv.iat[i, k])):
                                    # #print("Alpha Check",pd_csv.iat[i,k])
                                    row_alphabet_check = True
                                    break
                            if (row_alphabet_check):
                                # #print(pd_csv.iat[i-1,j])
                                pd_csv.iat[i, j] = pd_csv.iat[i - 1, j]

    # Identifying index of Principal
    i = 0
    for j in range(pd_csv.shape[1]):
        if ('Principal' in str(pd_csv.iat[i, j])):
            principal_index = j

    # Extracting the column data after principal
    pd_csv_principal = pd_csv.iloc[:, principal_index:]

    #Removing any (digit) values
    pd_csv_principal.iloc[1:, :] = pd_csv_principal.iloc[1:, :].replace(r"\(\d+\)", np.nan, regex=True)

    # Sweeping everyhing to left in extracted df
    pd_csv_principal = pd.read_csv(StringIO(re.sub(',+', ',', pd_csv_principal.to_csv())))
    pd_csv_principal = pd_csv_principal.iloc[:, 1:]

    # Dropping columns with Nulls
    pd_csv_principal = pd_csv_principal[pd_csv_principal.columns[pd_csv_principal.isnull().mean() < 0.75]]

    pd_csv_principal.replace('na', np.nan, inplace=True)
    # Pushing to right if only one null is present
    pd_csv_principal[pd_csv_principal.isnull().sum(axis=1) == 1] = pd_csv_principal[
        pd_csv_principal.isnull().sum(axis=1) == 1].shift(1, axis=1)

    pd_csv = pd_csv.iloc[:, :principal_index].reset_index()
    pd_csv = pd_csv.iloc[:, 1:]

    pd_csv = pd.concat([pd_csv, pd_csv_principal], axis=1)

    new_header = pd_csv.iloc[0]  # grab the first row for the header
    pd_csv = pd_csv[1:]  # take the data less the header row
    pd_csv.columns = new_header.str.replace(r"\(\d+\)", "")  # set the header row as the df header

    # Dropping columns which NaN as column names
    pd_csv = pd_csv.loc[:, pd_csv.columns.notna()]

    # Sweeping everyhing to left in extracted df
    #pd_csv = pd.read_csv(StringIO(re.sub(',+', ',', pd_csv_principal.to_csv(index=False))))
    list1=[]
    try:
        for col in pd_csv.columns:
            if not re.search('[a-zA-Z]', col) :
                print("column", col)
                #pd_csv = pd_csv.drop(col, axis=1)
                #pd_csv = pd_csv.drop(col, axis=1)
                list1.append(col)
            if col==".1":
                #print("column", col)
                list1.append(col)
            #pd_csv[col] = pd_csv[col].astype(str).str.replace(r"\(\d+\)", "", regex=True)
        for col in list1:
            pd_csv = pd_csv.drop(col, axis=1)

    except:
        print("Bracket digits not removed")

    # Identifying index of business Description
    business_desc_index = pd_csv.columns.get_loc("Business Description")
    business_desc_index = business_desc_index + 1

    ## Finding out rows which have columns other than till business description empty
    extra_index = pd_csv.index[pd_csv.isnull().sum(axis=1) == pd_csv.shape[1] - business_desc_index].to_list()
    pd_csv = pd_csv.drop(extra_index)
    # Finding out rows which have subtotal
    subtotal_index = pd_csv.index[pd_csv.isnull().sum(axis=1) == pd_csv.shape[1] - (business_desc_index + 2)].to_list()
    pd_csv = pd_csv.drop(subtotal_index)

    # Filling up NAs
    pd_csv = pd_csv.fillna("")
    #pd_csv.to_csv("gp3_Formated_"+qtr_date+".csv")
    pd_csv.to_excel(writer, index=None, header=True,sheet_name=qtr_date)
    #writer.save()


def scrape_data(date):
    #Extracting info from the excel
    outside = False
    inside = False
    data = pd.ExcelFile("HMS_fillings.xlsx")
    df = data.parse('HMS')
    url = df['html_link'].str.strip()
    date_filled = df['date_filed'].str.strip()
    date_filled.head()
    i = 0
    while(i<len(df)):
        #Extracting the content from the URL
        edgar_resp = get_content(url[i])
        edgar_soup = parse_and_trim(edgar_resp,"HTML")
        date_format = datetime.strptime(str(date_filled[i]),"%m/%d/%Y")
        print("Date of the doc",str(date_format))
        if(int(date_format.strftime("%Y"))>date):
            #print("Scraping Started")
            if(int(date_format.strftime("%m"))<=4):
                qtr_date = str("December 31, "+str(int(date_format.strftime("%Y"))-1))
            elif(int(date_format.strftime("%m"))<=5 and int(date_format.strftime("%m"))>3):
                qtr_date = str("March 31, "+str(int(date_format.strftime("%Y"))))
            elif(int(date_format.strftime("%m"))<=8 and int(date_format.strftime("%m"))>6):
                qtr_date = str("June 30, " + str(int(date_format.strftime("%Y"))))
            elif (int(date_format.strftime("%m")) <= 11 and int(date_format.strftime("%m")) > 9):
                qtr_date = str("September 30, " + str(int(date_format.strftime("%Y"))))
            print("QTR DATE",qtr_date)
            last_year = (int(date_format.strftime("%Y")) - 2)
            outside_string_check = "December 31, " + str(last_year)
            #print("OUTSIDE STRING CHECK", outside_string_check)
            print("OUTSIDE EXTRACTION")
            try:
                outside = extract_tables_outside(edgar_soup, qtr_date, i, outside_string_check)
            except:
                print("Outside Extract failed")
                outside = 0
            print("")
            print("INSIDE EXTRACTION")
            inside = extract_tables_inside(edgar_soup, qtr_date, i, outside_string_check)
            print("INSIDE", inside)
            print("OUTSIDE", outside)
            if (inside == False and outside == 0):
                print("Group 3 Extraction")
                extract_tables_3(edgar_soup, qtr_date, i)
                group3_formatting("output-" + str(i) + "-pandas.csv", qtr_date)
            else:
                try:
                    if(outside==2):
                        group3_formatting("output-" + str(i) + "-pandas.csv", qtr_date)
                    else:
                        formatting("output-" + str(i) + "-pandas.csv", qtr_date)
                except:
                    print("Error:::::::Table not found:::::::::::::")
        i = i+1

def extract_tables_outside(soup_content, qtr_date,ind,outside_string_check):
    master_table = None
    return_value = 0
    #### find tables when keyword is outside it
    for tag in soup_content.find_all(text=re.compile('^.*(Consolidated Schedule [Oo]f Investments|SCHEDULE OF INVESTMENTS|Schedule of Investments).*$')):
        #print("Tag",tag)
        date_str= ''
        tag_index = 0
        nextNode = tag
        try:
            while tag_index<3:
                nextNode = nextNode.find_next()
                #print("NextNode",nextNode)
                try:
                    if(qtr_date.lower() in unicodedata.normalize('NFKD', nextNode.text.strip()).lower()):
                        #print("Date Found")
                        date_str = unicodedata.normalize('NFKD', nextNode.text.strip())
                        break
                except AttributeError:
                    print("Error in Date String search")
                tag_index += 1
        except:
            print("No date_str found")
        #print("Date String",date_str)
        if qtr_date.lower() in date_str.lower():
            #print('Table found: ')
            ###### Use find_previous if the tag is td or tr.
            html_table = nextNode.findNext('table')
            ##print("HTML TABLE",html_table)
            if master_table is None:
                master_table = pd.read_html(html_table.prettify(), skiprows=0, flavor='bs4')[0]
                master_table = master_table.applymap(lambda x: unicodedata.normalize('NFKD', x.strip().strip(u'\u200b').replace('—','-')) if type(x) == str else x)
                master_table = master_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan,
                                                                                          regex=True)
                master_table = master_table.dropna(how='all', axis=0)
            else:
                new_table = pd.read_html(html_table.prettify(), skiprows=0, flavor='bs4')[0]
                new_table = new_table.applymap(lambda x: unicodedata.normalize('NFKD', x.strip().strip(u'\u200b').replace('—','-')) if type(x) == str else x)
                new_table = new_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan,
                                                                                          regex=True)
                new_table = new_table.dropna(how='all', axis=0)
                ##print(new_table.head())
                master_table = master_table.append(
                    new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
                    ignore_index=True)
    if master_table is not None:
        master_table = master_table.applymap(lambda x: x.strip().strip(u'\u200b') if type(x) == str else x)
        master_table = master_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan, regex=True).replace(r'^\s*\)\s*$', np.nan, regex=True)
        master_table = master_table.replace("~", ",", regex=True)
        ## Code to understand if keyword is inside table or outside
        # master_table.to_csv("output-" + str(ind) + "-outside-pandas.csv")
        for col in master_table.columns:
            if (master_table[col].astype(str).str.contains(outside_string_check).any() == True):
                return 0
            else:
                return_value = 1
        for i in range(3):
            if (master_table.iloc[i].astype(str).str.contains("Maturity Date").any() == True):
                master_table.to_csv("output-" + str(ind) + "-pandas.csv")
                return 2
        if(return_value == 1):
            master_table.to_csv("output-" + str(ind) + "-pandas.csv")
            return 1
    else:
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
        #master_table.to_csv("output-" + str(ind) + "inside-pandas.csv")
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

def extract_tables_3(soup_content, qtr_date,ind):
    master_table = None
    # If portfolio not there at top then don't merge it.
    #### find tables when keyword is outside it
    for tag in soup_content.find_all(text=re.compile('^.*(Consolidated Schedule [Oo]f Investments|SCHEDULE OF INVESTMENTS|Schedule of Investments).*$')):
        #print("Tag",tag)
        date_str= ''
        tag_index = 0
        nextNode = tag
        try:
            while tag_index<3:
                nextNode = nextNode.find_next()
                #print("NextNode",nextNode)
                try:
                    if(qtr_date.lower() in unicodedata.normalize('NFKD', nextNode.text.strip()).lower()):
                        #print("Date Found")
                        date_str = unicodedata.normalize('NFKD', nextNode.text.strip())
                        break
                except AttributeError:
                    print("Error in Date String search")
                tag_index += 1
        except:
            print("No date_str found")
        #print("Date String",date_str)
        if qtr_date.lower() in date_str.lower():
            #print('Table found: ')
            ###### Use find_previous if the tag is td or tr.
            html_table = nextNode.findNext('table')
            #print("HTML TABLE",html_table)
            if master_table is None:
                master_table = pd.read_html(html_table.prettify(), skiprows=0, flavor='bs4')[0]
                master_table = master_table.applymap(lambda x: unicodedata.normalize('NFKD', x.strip().strip(u'\u200b').replace('—','-')) if type(x) == str else x)
                master_table = master_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan,
                                                                                          regex=True)
                master_table = master_table.dropna(how='all', axis=0)
                #print(master_table)
                #master_table.to_csv("master_table_"+str(qtr_date)+str(tag_index)+".csv")
            else:
                new_table = pd.read_html(html_table.prettify(), skiprows=0, flavor='bs4')[0]
                new_table = new_table.applymap(lambda x: unicodedata.normalize('NFKD', x.strip().strip(u'\u200b').replace('—','-')) if type(x) == str else x)
                new_table = new_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan,
                                                                                          regex=True)
                new_table = new_table.dropna(how='all', axis=0)
                new_table.to_csv("New_Table_testing.csv")
                # master_table = master_table.append(new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
                #             ignore_index=True)
                try:
                    new_table_cols = new_table.columns[0][1]
                    #print(new_table_cols)
                    if('Portfolio Company' in new_table_cols):
                        #print("Table Added",new_table.head())
                        master_table = master_table.append(
                            new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
                            ignore_index=True)
                except:
                    print("Last table error")
    if master_table is not None:
        master_table = master_table.applymap(lambda x: x.strip().strip(u'\u200b') if type(x) == str else x)
        master_table = master_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan, regex=True).replace(r'^\s*\)\s*$', np.nan, regex=True)
        master_table = master_table.replace("~", ",", regex=True)
        master_table.to_csv("output-"+str(ind)+"-pandas.csv")
        return True
    else:
        return False
    #return html_table

scrape_data(2012)
writer.save()
