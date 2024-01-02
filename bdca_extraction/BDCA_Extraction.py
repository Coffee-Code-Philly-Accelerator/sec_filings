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
writer = pd.ExcelWriter('BDCA_Final_Test.xlsx')


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
    for col in pd_csv.columns:
        pd_csv[col] = pd_csv[col].astype(str).str.replace('  ', ' ')
        portfolio_index_list = pd_csv.index[
            pd_csv[col].astype(str).str.contains("Company") | pd_csv[col].astype(str).str.contains(
                "Investment")].to_list()
        if (len(portfolio_index_list) > 0):
            break
    portfolio_index = portfolio_index_list[0]
    # if(len(portfolio_index_list)>1):
    #     pd_csv.drop(portfolio_index_list[1], inplace=True)

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

    print("Portofolio", portfolio_index)
    # Removing rows before portfolio index
    if (pd_csv.iloc[portfolio_index].str.contains('Company').any() == True):
        pd_csv = pd_csv.iloc[portfolio_index:]

    # Identifying levels in the dataframe
    listOfLevels = list()
    for i in range(pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            if (pd.notnull(pd_csv.iat[i, j])):
                # print(pd_csv.iat[i, j])
                if ((('investments' in str(pd_csv.iat[i, j])) or ('Investments' in str(pd_csv.iat[i, j]))) and
                        ('Subtotal' not in pd_csv.iat[i, j]) and ('Total' not in pd_csv.iat[i, j]) and
                        ('Inc' not in pd_csv.iat[i, j]) and ('Fund' not in pd_csv.iat[i, j]) and
                        ('LLC' not in pd_csv.iat[i, j]) and ('L.P' not in pd_csv.iat[i, j]) and
                        ('CMS' not in pd_csv.iat[i, j]) and ('Holdings' not in pd_csv.iat[i, j])):
                    listOfLevels.append(i)


    listOfInvestments = list()

    if ((pd_csv.iloc[0].str.contains('Type of Investment').any() == False) and
            (pd_csv.iloc[0].str.contains('Type  of Investment').any() != True)):
        for i in range(pd_csv.shape[0]):
            for j in range(4):
                if (pd.notnull(pd_csv.iat[i, j])):
                    # print(pd_csv.iat[i, j])
                    if ((('Lien' in str(pd_csv.iat[i, j])) or
                         ('Senior Secured Debt' in str(pd_csv.iat[i, j])) or
                         ('Senior Unsecured Debt' in str(pd_csv.iat[i, j])) or
                         ('Subordinated Debt' in str(pd_csv.iat[i, j])) or
                         'Collateralized Securities' in str(pd_csv.iat[i, j]) or
                         'Equity/Other' in str(pd_csv.iat[i, j])
                    )
                            and (('Total' not in str(pd_csv.iat[i, j]))) and  ('Inc' not in pd_csv.iat[i, j])
                    and ('LLC' not in pd_csv.iat[i, j])):
                        listOfInvestments.append(i)
        # Adding the Levels to an array
        InvestmentValues = []
        InvestmentValues.append("Type of Investments")
        # InvestmentValues.append("")
        for i in range(len(listOfInvestments)):
            beginIndex = listOfInvestments[i]
            if (i == len(listOfInvestments) - 1):
                endIndex = len(pd_csv)
            else:
                endIndex = listOfInvestments[i + 1]
            value = pd_csv.iat[beginIndex, 0]
            value = ''.join([i for i in value if not i.isdigit()])
            value = re.sub("[^\P{P}-]+", "", value)
            for x in range(beginIndex, endIndex):
                InvestmentValues.append(value)
                # Adding the array to the df
        if (len(InvestmentValues) > 2):
            pd_csv.insert(1, 'Type of Investments', InvestmentValues)

        # Appending to Type of Investment from Company column

        type_invest_append = ['(Revolver)', '(First Out)', '(Last Out)', 'Series A', 'Series B', 'Series C',
                              'Series A-1',
                              'Series B-1', 'Class A', 'Class C', 'Class D', 'Membership Unit Warrants']

    else:
        print("Type of Investment not present")


    remove = string.punctuation
    remove = remove.replace("-", "")  # don't remove hyphens
    pattern = r"[{}]".format(remove)  # create the pattern

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
        value = ''.join([i for i in value if not i.isdigit()])
        value = re.sub("[^\P{P}-]+", "", value)
        print("Begin", beginIndex)
        print("End", endIndex)
        for x in range(beginIndex, endIndex):
            newValues.append(value)

    # print(newValues)
    if (len(newValues) < (len(pd_csv))):
        newValues.insert(0, 'Levels')
    # Adding the array to the df
    #pd_csv.insert(0, 'Levels', newValues)

    # Finding Subtotals
    pd_csv.reset_index()
    subtotal_list = list()
    total_list = list()
    for i in range(pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            if (('Subtotal' in str(pd_csv.iat[i, j]))):
                # print(pd_csv.iloc[i,:])
                subtotal_list.append(i)

    print("Subtotal", subtotal_list)
    # Dropping the subtotal rows
    pd_csv.drop(pd_csv.index[subtotal_list], inplace=True)

    # Finding Totals
    total_list = list()
    for i in range(pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            if ('Total' in str(pd_csv.iat[i, j])):
                total_list.append(i)

    print("Total list", total_list)
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
            elif ('Industry' in str(pd_csv.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                type_of_investment_index = j
    print("type_of_investment_index", type_of_investment_index)

    for i in range(5):
        for j in range(pd_csv.shape[1]):
            if ('Industry' in str(pd_csv.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                principal_index = j
                break
    print("Principal Index", principal_index)

    # Identifying index of maturity
    i = 0
    # Extracting the column data after shares/warrants/principal
    try:
        pd_csv_principal = pd_csv.iloc[:, principal_index:]
        pd_csv_principal = pd_csv_principal.replace("%", np.nan)
        pd_csv_principal = pd_csv_principal.replace("\u200b", np.nan)
        pd_csv_principal = pd_csv_principal.replace("\u200b.1", np.nan)
        pd_csv_principal = pd_csv_principal.replace("CAD", np.nan)

        pd_csv_principal = pd.read_csv(StringIO(re.sub(',+', ',', pd_csv_principal.to_csv())))

    except:
        print("no Maturity")

    pd_csv_principal = pd_csv_principal.iloc[:, 1:]
    # Dropping columns with Nulls
    perc = 95  # Like N %
    min_count = int(((100 - perc) / 100) * pd_csv_principal.shape[0] + 1)
    pd_csv_principal = pd_csv_principal.dropna(axis=1,
                                               thresh=min_count)

    for i in range(5):
        for j in range(pd_csv_principal.shape[1]):
            print(i, j)
            print(str(pd_csv_principal.iat[i, j]).translate({ord(c): None for c in string.whitespace}))
            try:
                if ('NetAssets' in str(pd_csv_principal.iat[i, j]).translate(
                        {ord(c): None for c in string.whitespace})):
                    net_assets_index = j
                    print("Net Assets Index", net_assets_index)
            except:
                print("Net Assets not present")
            if ('FairValue' in str(pd_csv_principal.iat[i, j]).translate({ord(c): None for c in string.whitespace})):
                fair_index = j
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

    try:
        principal_cols = pd_csv_principal.columns[principal_index]
        last_1_nan_list = pd_csv_principal.index[
            (pd_csv_principal.iloc[:, -1].isnull()) & (pd_csv_principal.iloc[:, -2].notnull()) & (
                pd_csv_principal.iloc[:, -3].notnull())].to_list()
        pd_csv_principal.loc[last_1_nan_list, principal_cols:] = pd_csv_principal.loc[last_1_nan_list,
                                                                 principal_cols:].shift(1, axis=1)
    except:
        print("No Rate")

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

    # Dropping columns which NaN as column names
    pd_csv = pd_csv.loc[:, pd_csv.columns.notna()]
    maturity_cols = [col for col in pd_csv.columns if 'Maturity' in col]
    pd_csv = pd_csv.dropna(thresh=len(pd_csv.columns) - 2)

    perc = 95  # Like N %
    min_count = int(((100 - perc) / 100) * pd_csv.shape[0] + 1)
    pd_csv = pd_csv.dropna(axis=1, thresh=min_count)

    last_nan = list()
    pd_csv = pd_csv.reset_index(drop=True)

    try:
        for i in range(int(pd_csv.shape[0])):
            if (pd_csv.iat[i, pd_csv.shape[1] - 1] == ' ' or pd_csv.iat[i, pd_csv.shape[1] - 1] == '  '):
                last_nan.append(i)
                pd_csv.iat[i, pd_csv.shape[1] - 1] = np.nan
        pd_csv.loc[last_nan, maturity_cols[0]:] = pd_csv.loc[last_nan, maturity_cols[0]:].shift(1, axis=1)
    except:
        print("Error while switching for last column")

    # Appending to Type of Investment from Company column

    type_invest_append = ['(Revolver)', '(First Out)', '(Last Out)', 'Series A', 'Series B', 'Series C', 'Series A-1',
                          'Series B-1', 'Class A', 'Class C', 'Class D', 'Membership Unit Warrants']

    try:
        comp_cols = [col for col in pd_csv.columns if 'Investment ,' in col]
        type_inv_cols = [col for col in pd_csv.columns if 'Type of Investment' in col]

        invest_append_list = list()
        for i in type_invest_append:
            print(i)
            if (pd_csv[comp_cols[0]].str.contains(i).any() == True):
                invest_append_list = pd_csv.index[pd_csv[comp_cols[0]].str.contains(i)].to_list()
                for j in invest_append_list:
                    if (i in str(pd_csv.loc[int(j), comp_cols[0]])):
                        pd_csv.loc[int(j), comp_cols[0]] = pd_csv.loc[int(j), comp_cols[0]].replace(i, "")
                        pd_csv.loc[int(j), type_inv_cols[0]] = type_inv_cols[0] + " " + i
    except:
        print("Company Name column different")

    try:
        pd_csv.drop('​.1', inplace=True, axis=1)
    except:
        print(".1 column not present")

    second_header = list()
    # Identifying where the second header is
    for index, row in pd_csv.iterrows():
        if index == 1:
            continue
        if 'Portfolio Company' in row[0]:
            second_header.append(index)
    # Dropping the second header
    try:
        print("Second Header", second_header)
        pd_csv.drop(second_header, inplace=True)
    except:
        print("No second header")

    maturity_cols = [col for col in pd_csv.columns if 'Maturity' in col]

    maturity_cols = [col for col in pd_csv.columns if 'Maturity' in col]

    try:
        last_2_nan_list = pd_csv.index[(pd_csv.iloc[:, -1].isnull()) & (pd_csv.iloc[:, -2].isnull()) & (
            pd_csv.iloc[:, -3].notnull())].to_list()
        pd_csv.loc[last_2_nan_list, maturity_cols[0]:] = pd_csv.loc[last_2_nan_list, maturity_cols[0]:].shift(2, axis=1)
    except:
        print("Last 2 Columns")

    # Shifting if Number of Shares is empty

    try:
        share_cols = [col for col in pd_csv.columns if 'Shares' in col]
        yield_cols = [col for col in pd_csv.columns if 'Yield' in col]

        share_nan_list = pd_csv.index[pd_csv.iloc[:, -4].isnull()].to_list()
        pd_csv.loc[share_nan_list, [share_cols[0], yield_cols[0]]] = pd_csv.loc[
            share_nan_list, [yield_cols[0], share_cols[0]]].values
    except:
        print("Last 2 Columns")
    # Shifting rows where cost is null
    try:
        maturity_cols = [col for col in pd_csv.columns if 'Maturity' in col]
        cost_cols = [col for col in pd_csv.columns if 'Cost' in col]
        cost_nan_list = pd_csv.index[pd_csv.loc[:, cost_cols[0]].isnull()].to_list()
        pd_csv.loc[cost_nan_list, maturity_cols[0]:cost_cols[0]] = pd_csv.loc[cost_nan_list,
                                                                   maturity_cols[0]:cost_cols[0]].shift(1, axis=1)
    except:
        print("Columns not present")
    pd_csv = pd_csv.fillna("")
    try:
        for i in range(int(pd_csv.shape[0])):
            for j in range(pd_csv.shape[1]):
                pd_csv.iat[i, j] = re.sub(r"\(\d+\)", "", pd_csv.iat[i, j])
    except:
        print("Bracket digits not removed")




    print("Formated Data Frame")
    print(pd_csv)
    print("Length",len(pd_csv))
    # pd_csv.to_csv("Formated_" + qtr_date + ".csv")
    pd_csv.to_excel(writer, index=None, header=True, sheet_name=qtr_date)
    #writer.save()


def scrape_data(date):
    #Extracting info from the excel
    outside = False
    inside = False
    data = pd.ExcelFile("bdca_fillings.xlsx")
    df = data.parse('BDCA')
    url = df['html_link'].str.strip()
    date_filled = df['date_filed']
    #date_filled.head()
    i = 0
    while(i<len(df)):
        #Extracting the content from the URL
        edgar_resp = get_content(url[i])
        edgar_soup = parse_and_trim(edgar_resp, "HTML")

        date_format = datetime.strptime(str(date_filled[i]), "%Y-%m-%d %H:%M:%S")
        print("Date of the doc", str(date_format))
        print("Year", (date_format.strftime("%Y")))
        print("Month", (date_format.strftime("%m")))
        if(int(date_format.strftime("%Y"))>=date):
            #print("Scraping Started")
            if(int(date_format.strftime("%m"))<=4):
                qtr_date = str("December 31, "+str(int(date_format.strftime("%Y"))-1))
            elif(int(date_format.strftime("%m"))<=5 and int(date_format.strftime("%m"))>3):
                qtr_date = str("March 31, "+str(int(date_format.strftime("%Y"))))
            elif(int(date_format.strftime("%m"))<=8 and int(date_format.strftime("%m"))>6):
                qtr_date = str("June 30, " + str(int(date_format.strftime("%Y"))))
            elif (int(date_format.strftime("%m")) <= 11 and int(date_format.strftime("%m")) > 9):
                qtr_date = str("September 30, " + str(int(date_format.strftime("%Y"))))
                
            last_year = (int(date_format.strftime("%Y")) - 2)
            outside_string_check = "December 31, " + str(last_year)
            print("OUTSIDE STRING CHECK", outside_string_check)
            print("OUTSIDE EXTRACTION")
            outside = extract_tables_outside(edgar_soup, qtr_date, i, outside_string_check)
            try:
                outside = extract_tables_outside(edgar_soup, qtr_date, i, outside_string_check)
                print("Outside Extraction Done")
            except:
                print("Outside Extract failed")
                outside = 0
            print("")
            print("INSIDE EXTRACTION")
            #inside = extract_tables_inside(edgar_soup, qtr_date, i, outside_string_check)
            print("INSIDE", inside)
            # print("OUTSIDE", outside)
            # if (inside == False and outside == 0):
            #     1
            #     # print("Group 3 Extraction")
            #     # extract_tables_3(edgar_soup, qtr_date, i)
            #     # group3_formatting("output-" + str(i) + "-pandas.csv", qtr_date)
            # else:
            #     #formatting("output-" + str(i) + "-pandas.csv", qtr_date)
            #     try:
            #         if(outside==2):
            #             group3_formatting("output-" + str(i) + "-pandas.csv", qtr_date)
            #         else:
            #             formatting_2("output-" + str(i) + "-pandas.csv", qtr_date)
            #     except:
            #         print("Error:::::::Table not found:::::::::::::")
            
            
            print("QTR DATE",qtr_date)
            formatting_2("output-" + str(i) + "-pandas.csv", qtr_date)
            try:
                formatting_2("output-" + str(i) + "-pandas.csv", qtr_date)
            except:
                print("DID NOT WORK FOR ",str(date_format))
        i = i+1

def extract_tables_outside(soup_content, qtr_date,ind,outside_string_check):
    master_table = None
    return_value = 0
    continous_table_check = 1
    #### find tables when keyword is outside it
    tag_search_test = '^.*(Consolidated Schedule [Oo]f Investments|CONSOLIDATED SCHEDULES OF INVESTMENTS|SCHEDULE OF INVESTMENTS|CONSOLIDATED SCHEDULE OF INVESTMENTS|hedule of Investments|Consolidated\nSchedule of Investments|CONSOLIDATED SCHEDULES\nOF INVESTMENTS).*$'
    for tag in soup_content.find_all(text=re.compile(tag_search_test)):
        print("Tag", tag)
        if("and" in tag):
            print("Initial Tag")
            continue
        date_str= ''
        tag_index = 0
        nextNode = tag
        table_in_tag = 0
        try:
            while tag_index<10:
                nextNode = nextNode.find_next()
                print("Next Node Next",nextNode)
                #print("NextNode",nextNode.text)
                if('Portfolio Company' in nextNode.text):
                    print("FOUND TABLE")
                    table_in_tag = 1
                    date_str = qtr_date.lower()
                    break
                if(qtr_date in nextNode.text):
                    print("Date in table")
                    date_str = qtr_date
                #if('')
                if(nextNode.text.strip() == 'outside_string_check'):
                    continous_table_check = 0
                    break
                if('Company' in str(nextNode)):
                    print("STRING CONTAINS TRUE")
                    print(tag.next_sibling.next_sibling.text.strip().lower())
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
        print("Date String",date_str)
        if (qtr_date.lower() in date_str.lower() and "for" not in date_str.lower()):
            ###### Use find_previous if the tag is td or tr.
            while(continous_table_check==1):
                if(table_in_tag == 0):
                    html_table = nextNode.findNext('table')
                else:
                    html_table = nextNode
                    table_in_tag = 0
                print("HTML TABLE",html_table)

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
                    try:
                        print('Total Investments, '+qtr_date)
                        print(' Total Portfolio Investments, '+qtr_date)
                        print("Last Row",new_table.iloc[-1,0].replace(" ", ""))
                        last_row = new_table.iloc[-1,0].replace(" ", "")
                        if('TOTALINVESTMENTS' in last_row):
                            print("Total Investment found")
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
                            print('TOTAL DERIVATIVES founds')
                            break
                        elif ('TotalInvestments' in last_row):
                            master_table = master_table.append(
                                new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
                                ignore_index=True)
                            continous_table_check = 0
                            print('TOTAL DERIVATIVES founds')
                            break
                        elif ('NETASSETS' in last_row):
                            master_table = master_table.append(
                                new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
                                ignore_index=True)
                            continous_table_check = 0
                            print('TOTAL DERIVATIVES founds')
                            break
                    except:
                        print("Total Investments not found")
                        print("Float object")
                    try:
                        for col in new_table.columns:
                            if (new_table[col].astype(str).eq("(1)").any() == False):
                                print("End not found")
                            else:
                                print("End of continous table")
                                continous_table_check = 0
                                break
                        if (continous_table_check == 1):
                            master_table = master_table.append(
                                new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
                                ignore_index=True)
                    except:
                        print("no new table")
                nextNode = nextNode.find_next('table')
                print("Last Next Node",nextNode)
                #nextNode = nextNode.findNext('table')
        if(continous_table_check == 0):
            print("BREAKING")
            break
    if master_table is not None:
        print("INSIDE NOT NONE")
        master_table = master_table.applymap(lambda x: x.strip().strip(u'\u200b') if type(x) == str else x)
        master_table = master_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan, regex=True).replace(r'^\s*\)\s*$', np.nan, regex=True)
        master_table = master_table.replace("~", ",", regex=True)
        ## Code to understand if keyword is inside table or outside
        # master_table.to_csv("output-" + str(ind) + "-outside-pandas.csv")
        count = 0
        for col in master_table.columns:
            try:
                count = count + 1
                if (master_table[col].astype(str).str.contains(outside_string_check).any() == True):
                    1
                    print("Outside String found")
                else:
                    #master_table.to_csv("output-" + str(ind) + "-pandas.csv")
                    return_value = 1
            except:
                print("not running full cols")
            if count > 2:
                break
        print("RETURN VALUE",return_value)
        if(return_value == 1):
            print("MASTER TABLE",master_table)
            master_table.to_csv("output-" + str(ind) + "-pandas.csv")
            return 1
    else:
        print("Master table none")
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

scrape_data(2012)
writer.close()
