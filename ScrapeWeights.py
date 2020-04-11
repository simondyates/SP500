# Scrapes the (difficult to get) weights for S&P500, NDX100 and Dow Jones Indices
# and saves to csv files

from bs4 import BeautifulSoup
import requests
import pandas as pd

def cust_replace(s):
# The website has an odd formatting for the rightmost column
   return s.replace('(', '').replace(')', '').replace('%', '')

indices = {'SPX': 'sp500', 'NDX': 'nasdaq100', 'DJX': 'dowjones'}
# my name, slickcharts name

for index in list(indices.keys()):
    response = requests.get('https://www.slickcharts.com/' + indices[index])
    text = BeautifulSoup(response.text, 'html.parser')

# Find the table rows in the html
    rowTags = text.find_all('tr')
    headerTags = rowTags[0].find_all('th')
    columns = [t.string.replace('\xa0', '') for t in headerTags]

# Parse them into a dataframe
    data = []
    for i in range(1, len(rowTags)):
        rowElements = [t for t in rowTags[i].find_all('td')]
        list = []
        for j in range(len(rowElements)):
            if j != 4:
                list.append(rowElements[j].string)
            else:
                list.append(rowElements[j].text.replace('\xa0', '').strip())
        data.append(list)

    data_df = pd.DataFrame(data, columns=columns)
    data_df.set_index('#', inplace=True)

# Convert to numeric
    for col in columns[3:-1]:
        data_df[col] = pd.to_numeric(data_df[col].str.replace(',', ''))

    data_df['% Chg'] = pd.to_numeric(data_df['% Chg'].apply(cust_replace))/100
    data_df['Weight'] = data_df['Weight']/100

# Find the as-of date
    p = text.find(lambda tag: tag.name=="p" and "Data as of" in tag.text)
    as_of_date = p.text[12:-2].replace('/', '-')

# Save csv
    data_df.to_csv(index + '_Weights_' + as_of_date + '.csv', index=False, float_format='%.5f')