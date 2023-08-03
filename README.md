# sec_filings

## Rough steps to get started
1. Create the sec filings links excel work book by using the links from sec website page for given cik
  a. www.sec.gov/edgar/searchedgar/companysearch
  b. Need to automate creation of weblinks and filing and reporting dates with form type
  c. Isolate th 10-q/10-k filings in the table and cik on each link to get the individual
2. Write python script to scrape the data from each quarterly filing (10-Q and 10-K)
3. With each scraped dataframe, clean up to the point that it represents the table as reported in the filing
4. Final file should be called some like clean_soi_data.csv
Final checks to make sure it works:
  a. Sum of cost and fair values should equal the “Total Investments” line at the bottom of the table.
  b. Sam can run the code on his end and generate
  c. Sam to review the cleaned up table visually
