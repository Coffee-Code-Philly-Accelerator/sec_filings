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
writer = pd.ExcelWriter('HMS_Final_Extraction_1.xlsx')


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


def get_rows_columns_map(table_result, blocks_map):
    rows = {}
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    if row_index not in rows:
                        # create new row
                        rows[row_index] = {}

                    # get the text value
                    rows[row_index][col_index] = get_text(cell, blocks_map)
    return rows


def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] == 'SELECTED':
                            text += 'X '
    return text

def generate_table_csv(table_result, blocks_map, table_index):
    rows = get_rows_columns_map(table_result, blocks_map)
    #print(rows)
    table_id = 'Table_' + str(table_index)

    # get cells.
    csv = 'Table: {0}\n\n'.format(table_id)

    for row_index, cols in rows.items():

        for col_index, text in cols.items():
            text = text.replace(",","~")
            print(text)
            csv += '{}'.format(text) + ","
        csv += '\n'
    #print(csv)
    csv += '\n\n\n'
    return csv

def get_table_csv_results(file_name):
    with open(file_name, 'rb') as file:
        img_test = file.read()
        bytes_test = bytearray(img_test)
        print('Image loaded', file_name)

    # process using image bytes
    # get the results
    client = boto3.client('textract')

    response = client.analyze_document(Document={'Bytes': bytes_test}, FeatureTypes=['TABLES'])

    # Get the text blocks
    blocks = response['Blocks']
    #pprint(blocks)

    blocks_map = {}
    table_blocks = []
    for block in blocks:
        blocks_map[block['Id']] = block
        if block['BlockType'] == "TABLE":
            table_blocks.append(block)

    if len(table_blocks) <= 0:
        return "<b> NO Table FOUND </b>"

    csv = ''
    for index, table in enumerate(table_blocks):
        csv += generate_table_csv(table, blocks_map, index + 1)
        csv += '\n\n'

    return csv

def formatting(file_name,qtr_date):
    pd_csv = pd.read_csv(file_name)
    # pd_csv.dropna(how='all', axis=1, inplace=True)
    # pd_csv = pd_csv.reset_index(drop=False)
    # print(pd_csv)
    # new_header = pd_csv.iloc[0]  # grab the first row for the header
    # pd_csv = pd_csv[1:]  # take the data less the header row
    # pd_csv.columns = new_header  # set the header row as the df header
    # pd_csv = pd_csv.replace("~", ",", regex=True)
    # new_col = pd_csv.loc[pd_csv.isnull().sum(axis=1) == len(pd_csv.iloc[0]) - 1].iat[0, 0]
    # pd_csv = pd_csv.dropna(thresh=len(pd_csv.columns) - 1)
    # pd_csv.insert(0, 'Levels', new_col)
    # file_name_string = file_name.split(".")[0]
    # Removing the first column of the csv
    pd_csv = pd_csv.iloc[:,1:]
    # Resetting index
    pd_csv.reset_index()
    # Find out where portfolio company is
    for col in pd_csv.columns:
        portfolio_index_list = pd_csv.index[pd_csv[col].astype(str).str.contains("Portfolio Company")].to_list()
        break
    portfolio_index = portfolio_index_list[0]
    # Removing rows before portfolio index
    pd_csv = pd_csv.iloc[portfolio_index:]
    # Adding company for the middle values where it might be missed out
    for i in range(2, pd_csv.shape[0]):
        for j in range(pd_csv.shape[1]):
            row_alphabet_check = False
            if (pd.isnull(pd_csv.iat[i, j])):
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
    # Pushing values to the left and all the NaN to the right
    pd_csv = pd.read_csv(StringIO(re.sub(',+', ',', pd_csv.to_csv())))
    perc = 100  # Like N %
    min_count = int(((100 - perc) / 100) * pd_csv.shape[0] + 1)
    #Dropping columns which have 100% NaN
    pd_csv = pd_csv.dropna(axis=1,
                           thresh=min_count)
    new_header = pd_csv.iloc[0]  # grab the first row for the header
    pd_csv = pd_csv[1:]  # take the data less the header row
    pd_csv.columns = new_header  # set the header row as the df header
    pd_csv.reset_index()
    pd_csv = pd_csv.iloc[:, 1:]
    pd_csv = pd_csv.replace("~", ",", regex=True)
    new_col = pd_csv.loc[pd_csv.isnull().sum(axis=1) == len(pd_csv.iloc[0]) - 1].iat[0, 0]
    pd_csv = pd_csv.dropna(thresh=len(pd_csv.columns) - 1)
    for col in pd_csv.columns:
        print("Column",col)
        pd_csv[col] = pd_csv[col].astype(str).str.replace(r"\(\d+\)", "",regex=True)
    pd_csv = pd_csv.fillna("")
    pd_csv.insert(0, 'Levels', new_col)
    for col in pd_csv.columns:
        pd_csv[col] = pd_csv[col].astype(str).str.replace(r"\(\d+\)", "")
    print(pd_csv)
    pd_csv.to_csv("Formated_"+qtr_date+".csv")
    pd_csv.to_excel(writer, index=None, header=True,sheet_name=qtr_date)
    writer.save()


def scrape_data(date):
    #Extracting info from the excel
    data = pd.ExcelFile("HMS_fillings.xlsx")
    df = data.parse('HMS')
    url = df['html_link']
    date_filled = df['date_filed']
    date_filled.head()
    i = 0
    while(i<2):
        #Extracting the content from the URL
        edgar_resp = get_content(url[i])
        edgar_soup = parse_and_trim(edgar_resp,"HTML")
        date_format = datetime.strptime(str(date_filled[i]),"%m/%d/%Y")
        print("Date of the doc",str(date_format))
        if(int(date_format.strftime("%Y"))>date):
            print("Scraping Started")
            if(int(date_format.strftime("%m"))<=4):
                qtr_date = str("December 31, "+str(int(date_format.strftime("%Y"))-1))
            elif(int(date_format.strftime("%m"))<=5 and int(date_format.strftime("%m"))>3):
                qtr_date = str("March 31, "+str(int(date_format.strftime("%Y"))))
            elif(int(date_format.strftime("%m"))<=8 and int(date_format.strftime("%m"))>6):
                qtr_date = str("June 30, " + str(int(date_format.strftime("%Y"))))
            elif (int(date_format.strftime("%m")) <= 11 and int(date_format.strftime("%m")) > 9):
                qtr_date = str("September 30, " + str(int(date_format.strftime("%Y"))))
            print("QTR DATE",qtr_date)
            table = extract_tables(edgar_soup,qtr_date,i)
            output = 'output' + str(i) + '.jpg'
            #Downloand the wkhtmltopdf folder and place it in the following location or wherever feasible and link the path to it.
            #imgkit.from_file('testing_'+str(i)+'.html', output,config = imgkit.config(wkhtmltoimage='C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltoimage.exe'))
            #formatting("output-" + str(i) + "-pandas.csv", qtr_date)
            try:
                formatting("output-" + str(i) + "-pandas.csv", qtr_date)
            except:
                print("Error:::::::Table not found:::::::::::::")
            # table_csv = get_table_csv_results(output)
            # output_file = 'output' + str(i) + '.csv'
            # # replace content
            # with open(output_file, "wt") as fout:
            #     fout.write(table_csv)
            # try:
            #     formatting(output_file,qtr_date)
            # except:
            #     print("Error:::::::Table not found:::::::::::::")
            # # show the results
            # print('CSV OUTPUT FILE: ', output_file)


            # if not table.empty:
            #     #table.to_csv('Check'+str(i)+'.csv')
            #     final_table = process_table(table,"HMS")
            #     print(final_table)
            # if not final_table.empty:
            #     final_table.to_excel(writer, qtr_date)
            #     writer.save()
            #     print("Saved to csv")
        i = i+1

def extract_tables(soup_content, qtr_date,ind):
    master_table = None
    #### Find all tables when the keyword is inside the tablee
    for tag in soup_content.find_all("table"):
        for tag_child in tag.find_all(text=re.compile('^.*(Consolidated Schedule [Oo]f Investments|SCHEDULE OF INVESTMENTS|Schedule of Investments).*$')):
            print("Tag Child",tag_child)
            for tag_children in tag.find_all(text=re.compile("As of "+qtr_date)):
                print("Tag Children",tag_children)
                if(qtr_date in tag_children):
                    master_table = pd.read_html(tag.prettify(), skiprows=0, flavor='bs4')[0]
                    master_table = master_table.applymap(
                        lambda x: unicodedata.normalize('NFKD', x.strip().strip(u'\u200b').replace('—', '-')) if type(
                            x) == str else x)
                    master_table = master_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan,
                                                                                              regex=True)
                    master_table = master_table.dropna(how='all', axis=0)
                    f = open('testing_tag' + qtr_date + '.html', 'w')
                    f.write(str(tag))
                    f.close()
    #### find tables when keyword is outside it
    # for tag in soup_content.find_all(text=re.compile('^.*(Consolidated Schedule [Oo]f Investments|SCHEDULE OF INVESTMENTS|Schedule of Investments).*$')):
    #     print("Tag",tag)
    #     date_str= ''
    #     tag_index = 0
    #     nextNode = tag
    #     try:
    #         while tag_index<3:
    #             nextNode = nextNode.find_next()
    #             print("NextNode",nextNode)
    #             try:
    #                 if(qtr_date.lower() in unicodedata.normalize('NFKD', nextNode.text.strip()).lower()):
    #                     print("Date Found")
    #                     date_str = unicodedata.normalize('NFKD', nextNode.text.strip())
    #                     break
    #             except AttributeError:
    #                 print("Error in Date String search")
    #             tag_index += 1
    #     except:
    #         print("No date_str found")
    #     print("Date String",date_str)
    #     if qtr_date.lower() in date_str.lower():
    #         print('Table found: ')
    #         ###### Use find_previous if the tag is td or tr.
    #         html_table = nextNode.findNext('table')
    #         print("HTML TABLE",html_table)
    #         f = open('testing_'+str(ind)+'.html', 'w')
    #         f.write(str(html_table))
    #         f.close()
    #         if master_table is None:
    #             master_table = pd.read_html(html_table.prettify(), skiprows=0, flavor='bs4')[0]
    #             master_table = master_table.applymap(lambda x: unicodedata.normalize('NFKD', x.strip().strip(u'\u200b').replace('—','-')) if type(x) == str else x)
    #             master_table = master_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan,
    #                                                                                       regex=True)
    #             master_table = master_table.dropna(how='all', axis=0)
    #         else:
    #             new_table = pd.read_html(html_table.prettify(), skiprows=0, flavor='bs4')[0]
    #             new_table = new_table.applymap(lambda x: unicodedata.normalize('NFKD', x.strip().strip(u'\u200b').replace('—','-')) if type(x) == str else x)
    #             new_table = new_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan,
    #                                                                                       regex=True)
    #             new_table = new_table.dropna(how='all', axis=0)
    #             #print(new_table.head())
    #             master_table = master_table.append(
    #                 new_table.dropna(how='all', axis=0).reset_index(drop=True).drop(index=0),
    #                 ignore_index=True)
    if master_table is not None:
        master_table = master_table.applymap(lambda x: x.strip().strip(u'\u200b') if type(x) == str else x)
        master_table = master_table.replace(r'^\s*$', np.nan, regex=True).replace(r'^\s*\$\s*$', np.nan, regex=True).replace(r'^\s*\)\s*$', np.nan, regex=True)
    master_table = master_table.replace("~", ",", regex=True)
    master_table.to_csv("output-"+str(ind)+"-pandas.csv")
    #return html_table

scrape_data(2012)
