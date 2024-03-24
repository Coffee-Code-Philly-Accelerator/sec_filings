import re
import pandas as pd

def main()->None:
    file_path = r'C:\Users\pysol\Desktop\projects\sec_filings\bdca_extraction\BDCA_Final_Test.xlsx'
    excel = pd.read_excel(file_path,sheet_name=None)

    dfs,footnotes = [],''
    for idx,date in enumerate(excel):
        df = excel[date]
        filtered_columns = df.filter(like='Portfolio Company')
        pc_name = filtered_columns.columns[0]
        pc,notes = pc_name.split('Portfolio Company') 
        footnotes += notes
        
        df['as_of_date'] = date
        df['order_within_filing'] = idx
        df['issuer'] = df[pc_name].str.upper()
        df.issuer = df.issuer.str.strip().str.replace(r'\s+', ' ', regex=True)
        df.issuer = df.issuer.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
        df.issuer = df.issuer.str.replace(r'\[\d+\]', '', regex=True).str.replace(r'\*+', '', regex=True)
        df.drop(pc_name,axis=1,inplace=True)
        
        filtered_columns = df.filter(like='Type of Investment')
        type_name = filtered_columns.columns[0]
        toi_name,notes = type_name.split('Type of Investment') 
        df.rename({type_name: toi_name},inplace=True)
        
        footnotes += notes
        pattern = r'\(.*?\)'
        df.columns = df.columns.str.replace(r'\s+','',regex=True)
        for col in df.columns: 
            col = col.strip()
            matches = re.findall(pattern,col)
            if len(matches) > 0:
                ocol = re.sub(pattern,'',col)
                footnotes += ''.join(matches)
                df.rename(columns={col:ocol},inplace=True)
        df['footnotes'] = footnotes
        footnotes = ''
        dfs.extend([df])
    df = pd.concat(dfs,axis=0)
    df.to_excel('processed.xlsx',index=False)

if __name__ == '__main__':
    main()
    