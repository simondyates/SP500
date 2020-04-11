from bs4 import BeautifulSoup
import requests
import pandas as pd

response = requests.get('https://www.slickcharts.com/sp500')
text = BeautifulSoup(response.text, 'html.parser')

rowTags = text.find_all('tr')
headerTags = rowTags[0].find_all('th')
columns = [t.string.replace('\xa0', '') for t in headerTags]

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

for col in columns[3:-1]:
    data_df[col] = pd.to_numeric(data_df[col].str.replace(',', ''))

def cust_replace(s):
   return s.replace('(', '').replace(')', '').replace('%', '')

data_df['% Chg'] = pd.to_numeric(data_df['% Chg'].apply(cust_replace))/100
data_df['Weight'] = data_df['Weight']/100

p = text.find(lambda tag: tag.name=="p" and "Data as of" in tag.text)
as_of_date = p.text[12:-2]