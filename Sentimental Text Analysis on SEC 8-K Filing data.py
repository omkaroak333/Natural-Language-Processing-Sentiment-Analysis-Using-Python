
import requests
import pandas
import numpy
import os
from sklearn import linear_model
import math
import seaborn
import statsmodels.api as sm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#Listing range fof years and quarters
lstYear = list(range(1995,2018))
lstQTR = list(range(1,5))

#Creating empty dataframe
csvData = pandas.DataFrame()

#Looping to get 8-K files
for year in lstYear:
    for qtr in lstQTR:
        print(str(year)+ ' - '+str(qtr))
        #Defining file path and request string
        strReq = 'https://www.sec.gov/Archives/edgar/full-index/' + str(year) + '/QTR' + str(qtr) + '/master.idx'
        filePath = str(year) + '/QTR' + str(qtr) + '/'

        #Creating directory if not already present
        if not os.path.exists(os.path.dirname(filePath)):
            os.makedirs(os.path.dirname(filePath))

        #Writing master.idx file
        file_write = open(filePath + 'master.idx','wb')
        file_write.write(requests.get(strReq).content)
        file_write.close()

        #Opening master.idx file and ignoring first 11 lines
        f_read = open(filePath + 'master.idx','r')
        lines_after_11 = f_read.readlines()[12:]

        #Creating empty dataframe
        data = pandas.DataFrame()

        #Getting 8-K detials
        for line in lines_after_11:
            lstStr = line.split('|')
            if(lstStr[2] == '8-K'):
                tempDF = pandas.DataFrame()
                tempDF.loc[0,'CIK'] = lstStr[0]
                tempDF.loc[0,'Date Filed'] = lstStr[3]
                tempDF.loc[0,'Filename'] = lstStr[4][:-1]
                data = pandas.concat([data,tempDF])

        #Sampling 100 dataset
        sampleData = data.sample(100)
        #Appending to main dataframe
        csvData = pandas.concat([csvData,sampleData])
        csvData['Date Filed'] = pandas.to_datetime(csvData['Date Filed'])

        #Getting the 8-K files
        for index,row in sampleData.iterrows():
            CIK = row['CIK']
            file_write = open(filePath + CIK + '_8K.txt','wb')
            file_write.write(requests.get('https://www.sec.gov/Archives/'+row['Filename']).content)
            file_write.close()


#Writing to CSV
csvData.to_csv('Track Data.csv',index= False)
csvData.to_csv('abc.csv',index= False)


#Reading merged DSF-DSI-FUNDA data
data = pandas.read_csv("Track Data.csv")
data = data.reset_index(drop = True)

#Creating empty dataframe
finalData = pandas.DataFrame()

counter = 1
for index,row in csvData.iterrows():
    print(counter)
    counter = counter + 1
    #Getting CIK and Date for each row
    cik = int(row['CIK'])
    date = row['Date Filed']

    #Subsetting data for the given CIK
    tempData = data.loc[data['CIK'] == cik]
    tempData['DATE'] = pandas.to_datetime(tempData['DATE'])
    tempData = tempData.reset_index(drop = True)


    if(len(tempData.index) != 0 ):

        #Getting closest date if 8-K filing date is not on a trading day
        if len(tempData.index[tempData['DATE'] == date]) == 0:
            date = tempData.iloc[(tempData['DATE']-date).abs().argsort()[:1]].iloc[0,2]

        #Computing variables
        filedDateIndex = tempData.index[tempData['DATE'] == date].tolist()[0]
        tempData['SHROUT'] = tempData['SHROUT'] * 1000
        tempData['Turnover'] = tempData['VOL'] / tempData['SHROUT']

        #Performing regression to calculate alpha and beta
        X = tempData.loc[:,'VWRETD']
        Y = tempData.loc[:,'RET']

        X2 = sm.add_constant(X)

        model = sm.OLS(Y,X2)
        results = model.fit()

        beta = results.params[1]

        alpha = results.params[0]

        for index,row in tempData.iterrows():
            tempData.loc[index,'logTOplusC'] = math.log10(row['Turnover'] + (2.55 / 1000000) )

        #Calculating Abnormal Returns and Turnovers
        for index,row in tempData.iterrows():
            tempData.loc[index,'Abnormal TurnOver'] = ( row['logTOplusC'] - numpy.average(tempData.iloc[tempData.index[tempData['DATE'] == date][0]+range(-71,-11)]['logTOplusC']) ) / numpy.std(tempData.iloc[tempData.index[tempData['DATE'] == date][0]+range(-71,-11)]['logTOplusC'])

            tempData.loc[index,'Abnormal Return'] = row['RET'] - (alpha + beta*row['VWRETD'])


        CAR = tempData.loc[filedDateIndex-5:filedDateIndex+6,:]
        CATO = tempData.loc[filedDateIndex-5:filedDateIndex+6,:]

        #Calculate cumulative abnormal returns and turnovers for multiple ranges
        CAR_1_1 = CAR.loc[4:7,'Abnormal Return'].sum()
        CAR_2_2 = CAR.loc[3:8,'Abnormal Return'].sum()
        CAR_3_3 = CAR.loc[2:9,'Abnormal Return'].sum()
        CAR_4_4 = CAR.loc[1:10,'Abnormal Return'].sum()
        CAR_5_5 = CAR['Abnormal Return'].sum()

        CATO_1_1 = CATO.loc[4:7,'Abnormal TurnOver'].sum()
        CATO_2_2 = CATO.loc[3:8,'Abnormal TurnOver'].sum()
        CATO_3_3 = CATO.loc[2:9,'Abnormal TurnOver'].sum()
        CATO_4_4 = CATO.loc[1:10,'Abnormal TurnOver'].sum()
        CATO_5_5 = CATO['Abnormal TurnOver'].sum()

        tempFinalData = pandas.DataFrame()
        tempFinalData.loc[0,'CAR_1_1'] = round(CAR_1_1,5)
        tempFinalData.loc[0,'CAR_2_2'] = round(CAR_2_2,5)
        tempFinalData.loc[0,'CAR_3_3'] = round(CAR_3_3,5)
        tempFinalData.loc[0,'CAR_4_4'] = round(CAR_4_4,5)
        tempFinalData.loc[0,'CAR_5_5'] = round(CAR_5_5,5)
        tempFinalData.loc[0,'CATO_1_1'] = round(CATO_1_1,5)
        tempFinalData.loc[0,'CATO_2_2'] = round(CATO_2_2,5)
        tempFinalData.loc[0,'CATO_3_3'] = round(CATO_3_3,5)
        tempFinalData.loc[0,'CATO_4_4'] = round(CATO_4_4,5)
        tempFinalData.loc[0,'CATO_5_5'] = round(CATO_5_5,5)

        finalData = pandas.concat([finalData,tempFinalData])


#Plotting CARs and CATOs
ax = seaborn.kdeplot(finalData['CAR_1_1'])
fig1 = ax.get_figure()
fig.savefig('CAR_1_1.png')
ax = seaborn.distplot(finalData['CAR_2_2'])
fig = ax.get_figure()
fig.savefig('CAR_2_2.png')
ax = seaborn.distplot(finalData['CAR_3_3'])
fig = ax.get_figure()
fig.savefig('CAR_3_3.png')
ax = seaborn.distplot(finalData['CAR_4_4'])
fig = ax.get_figure()
fig.savefig('CAR_4_4.png')
ax = seaborn.distplot(finalData['CAR_5_5'])
fig = ax.get_figure()
fig.savefig('CAR_5_5.png')

ax = seaborn.distplot(finalData['CATO_1_1'])
fig = ax.get_figure()
fig.savefig('CATO_1_1.png')
ax = seaborn.distplot(finalData['CATO_2_2'])
fig = ax.get_figure()
fig.savefig('CATO_2_2.png')
ax = seaborn.distplot(finalData['CATO_3_3'])
fig = ax.get_figure()
fig.savefig('CAR_1_1.png')
ax = seaborn.distplot(finalData['CATO_4_4'])
fig = ax.get_figure()
fig.savefig('CATO_4_4.png')
ax = seaborn.distplot(finalData['CATO_5_5'])
fig = ax.get_figure()
fig.savefig('CATO_5_5.png')

print(finalData.describe())

#Exporting Descriptive Statistics
finalData.describe().loc[['mean','std','min','25%','50%','75%','max'],:].to_csv('Descriptive Statistics.csv')





