#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import from Excel:
df = pd.read_excel('CAPM.xls', header = 0, index_col = 0)

#Covert to numpy array:
df.dtypes
df = df[2:].astype(float)
data = df.to_numpy()
df.dtypes

#Divide data into column-vectors:
ford_prices = data[:,0]
ibm_prices = data[:,1]
market_index = data[:,2]

riskfree_rate = data[:,3]/100

#Basic Plotting:
plt.figure(1)
plt.figure(figsize=[16,10])
plt.title('Stock Performance of IBM and FORD (monthly rates)')
plt.ylabel('USD')
plt.xlabel('Date')
plt.plot(df['FORD MOTOR'])
plt.plot(df['IBM'])
plt.legend(df.columns)
plt.show()

#Plotting with dates on x-axis
plt.figure(2)
plt.figure(figsize=[16,10])
plt.title('Stock Performance of IBM and FORD (monthly rates)')
plt.ylabel('USD')
plt.xlabel('Date')
plt.plot(df['FORD MOTOR'])
plt.plot(df['IBM'])
ticks = df.index[np.arange(0,len(df.index),12)] #use tick function to define ticks from 0 to length of index, taking every 12 data
plt.xticks(ticks)
plt.legend(df.columns)
plt.show()

#number of observations:
n = len(ford_prices)
print('number of observations: ', n)

y = round(n/12)
print('number of years: ', y)

#Calculating the rates of return (monthly):
ford_return = ((ford_prices[1:]/ford_prices[:-1])-1)
ibm_return = ((ibm_prices[1:]/ibm_prices[:-1])-1)
market_return = ((market_index[1:]/market_index[:-1])-1)

#Scatter-Plot: Correlation between Market return and risk-free rate:
plt.figure(3)
plt.scatter((riskfree_rate[:-1]+1)**(1/12)-1, market_return)
plt.title('Market return (monthly) vs. Risk Free Rate (adapted monthly)')
plt.xlabel('r_f')
plt.ylabel('r_M')
plt.xlim([3.5*10**-3, 5*10**-3])
plt.show()

#Plot Correlation between FORD and Market, and IBM and Market:
plt.figure(4, figsize=[16,10])
plt.subplot(121)
plt.scatter(market_return, ford_return)
plt.title('Market return versus FORD stock return')
plt.grid()

plt.subplot(122)
plt.scatter(market_return, ibm_return)
plt.title('Market return versus IBM stock return')
plt.grid()
plt.show()


#Betas of stocks:
def get_beta(stock_return, market_return, n):
    """
    Returns the beta of an asset
    
    Parameters
    ----------
    stock_return (numpy array): monthly rates of return of a stock
    market_return (numpy array): monthly rates of return of market
    n (int): number of observations
    
    Returns
    -------
    float: the beta coefficient of that stock
    """
    stock_expected_return = np.mean(stock_return)
    market_expected_return = np.mean(market_return)
    var_market = np.var(market_return)
    cov = np.sum((market_return - market_expected_return)*
                             (stock_return - stock_expected_return))/(n-2)
    return cov / var_market
    
beta_ford = get_beta(ford_return, market_return, n)
beta_ibm = get_beta(ibm_return, market_return, n)

print(beta_ford)
print(beta_ibm)

#Annualized rates:
a_expected_return_ibm = ((1+np.mean(ibm_return))**12)-1
a_expected_return_ford = ((1+np.mean(ford_return))**12)-1
a_expected_return_market = ((1+np.mean(market_return))**12)-1

#CAPM Graphical Illustration
average_risk_free_rate = np.mean(riskfree_rate[:n-1])
beta = np.arange(0,2.5,0.01)
security_line = average_risk_free_rate+beta*(a_expected_return_market -                                             average_risk_free_rate)
plt.figure(5)
plt.figure(figsize=[16,10])
plt.plot(beta,security_line)
plt.plot(beta_ibm, a_expected_return_ibm, 'ko')
plt.plot(beta_ford,a_expected_return_ford, 'ro')
plt.title('Capital Asset Pricing Model, Annual Returns')
plt.legend(['CAPM', 'IBM', 'FORD'])
plt.xlabel('Beta')
plt.ylabel('Expected Return')
plt.show()


# In[ ]:




