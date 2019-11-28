
import requests
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt - is not working for MAC OS X, need to add:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#getCurrencyAndDate function which return currencyCource
def getCurrencyAndDate(currency, dateFrom, dateTo):
    getDataFromUrl = requests.get('http://api.nbp.pl/api/exchangerates/rates/A/'+currency+'/'+dateFrom+'/'+dateTo).json()
    currencyCource = pd.DataFrame.from_dict(getDataFromUrl['rates'])
    currencyCource['effectiveDate'].astype('datetime64')
    currencyCource.set_index('effectiveDate', inplace=True)
    return currencyCource

#creating usdTable and eurTable
usdTable = getCurrencyAndDate('USD','2019-09-01','2019-09-30')
eurTable = getCurrencyAndDate('EUR','2019-09-01','2019-09-30')

#printing first 7 rows of the tables
print('------currency usd-------')
print(usdTable.head(7))
print('------currency eur-------')
print(eurTable.head(7))

#printing .info() and .describe() methods for USD course
print()
print('----- info() and describe() for USD course ------')
print(usdTable.info())
print(usdTable.describe())

#printing .info() and .describe() methods for EUR course
print()
print('----- info() and describe() for EUR course ------')
print(eurTable.info())
print(eurTable.describe())

#cleaning: deleting no
usdTable = getCurrencyAndDate('USD','2019-09-01','2019-09-30')
eurTable = getCurrencyAndDate('EUR','2019-09-01','2019-09-30')
usdTableCleaned = usdTable['mid'].head(7)
eurTableCleaned = eurTable['mid'].head(7)

#printing cleaned tables
print()
print('---- usdTable after deleting no ---- ')
print(usdTableCleaned)
print()
print('---- eurTable after deleting no ---- ')
print(eurTableCleaned)

#printing chart of usdTableCleaned
plt.plot(usdTableCleaned)
print('---- the chart of usd currency ----')
plt.show()

#printing chart of eurTableCleaned
plt.plot(eurTableCleaned)
print('---- the chart of eur currency ----')
plt.show()

#printing correlation of usdTableCleaned
print('---- the chart with correlation of courses -----')
print(np.corrcoef(usdTableCleaned, usdTableCleaned))

#drawing 2 graphs of courses
draw, (drawUsd, drawEur) = plt.subplots(2, sharex=True)
draw.suptitle('orange - USD line , green - EUR line')
drawUsd.plot(usdTableCleaned, 'tab:orange')
drawEur.plot(usdTableCleaned, 'tab:green')
plt.show()

