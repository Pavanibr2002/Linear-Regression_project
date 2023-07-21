# import libraries
import pandas as pd
import matplotlib.pyplot as plt

#read data into a dataframe
data=pd.read_csv('advertising.csv')
data.head()

#print the shape of the data frame
data.shape

#data visualisation
fig,axs=plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(16,8))
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[2])

#create X and y
feature_cols=['TV']
X=data[feature_cols]
y=data.Sales

#follow the usual sklearn pattern: import,instantiate
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X,y) #using values of X and y model is trained.

#print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)

#manually calculate the prediction
6.97482+0.055464*50

X_new=pd.DataFrame({'TV':[50]})
X_new.head()
lm.predict(X_new)

#create a dataframe with a minimum and maximum values of Tv
X_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

preds=lm.predict(X_new)
preds

data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(X_new,preds,c='red',linewidth=2)

import statsmodels.formula.api as smf
lm=smf.ols(formula='Sales ~ TV',data=data).fit()
lm.conf_int() #confidence interval between sales and tv

lm.pvalues

# r squared=least distance squared ,lies b/w 0 and 1
#higher the rsquared value then better the fit.
lm.rsquared

#multiple linear regression
feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
y=data.Sales

lm=LinearRegression()
lm.fit(X,y)

print(lm.intercept_)
print(lm.coef_)

lm=smf.ols(formula='Sales ~ TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()

#only include tv and radio in the model
lm=smf.ols(formula='Sales ~ TV+Radio',data=data).fit()
lm.rsquared

#handling categorical variables
import numpy as np

#set a seed for reproducability
np.random.seed(12345)

nums=np.random.rand(len(data))
mask_large=nums>0.5

data['Size']='small'
data.loc[mask_large,'Size']='Large'
data.head()

data['IsLarge']=data.Size.map({'small':0,'Large':1})
data.head()

feature_cols=['TV','Radio','Newspaper','IsLarge']
X=data[feature_cols]
y=data.Sales

lm= LinearRegression()
lm.fit(X,y)

print(feature_cols,lm.coef_)

#handling categorical data with more than 2 categories

#set a seed for reproducability
np.random.seed(123456)

#assign roughly 1/3rd of the observations to each group
nums=np.random.rand(len(data))
mask_suburban=(nums>0.33)&(nums<0.66)
mask_urban=nums>0.66
data['Area']='rural'
data.loc[mask_suburban,'Area']='suburban'
data.loc[mask_urban,'Area']='urban'
data.head()

area_dummies=pd.get_dummies(data.Area,prefix='Area').iloc[:,1:]
area_dummies

data=pd.concat([data,area_dummies],axis=1)
data.head()

feature_cols=['TV','Radio','Newspaper','IsLarge','Area_suburban','Area_urban']
X=data[feature_cols]
y=data.Sales

lm=LinearRegression()
lm.fit(X,y)

print(feature_cols,lm.coef_)