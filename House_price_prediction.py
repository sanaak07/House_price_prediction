#!/usr/bin/env python
# coding: utf-8

# Problem Statement:
# 
# A US-based housing company named Surprise Housing has decided to enter the Australian market. The company uses data analytics to purchase houses at a price below their actual values and flip them on at a higher price. For the same purpose, the company has collected a data set from the sale of houses in Australia. The data is provided in the CSV file below. The company is looking at prospective properties to buy to enter the market. You are required to build a regression model using regularisation in order to predict the actual value of the prospective properties and decide whether to invest in them or not. The company wants to know the following things about the prospective properties:
# 
# * Which variables are significant in predicting the price of a house, and how well those variables describe the price of a house.
# * Also, determine the optimal value of lambda for ridge and lasso regression.

# In[1]:


## Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sweetviz as sv
import warnings
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[2]:


pip install sweetviz


# In[3]:


warnings.filterwarnings('ignore')


# # Reading and Understanding the Data

# In[4]:


# Reading data from csv
housing_df= pd.read_csv('train.csv')
housing_df.head()


# In[5]:


# Checking shape
housing_df.shape


# In[6]:


# Checking dataframe info
housing_df.info()


# In[7]:


# Checking descriptive statistics
housing_df.describe().T


# # Basic Data Cleanup

# # Missing value check and imputation using Business Logic

# In[8]:


## Checking percentage of missing values
missing_info= round(housing_df.isna().sum() * 100/housing_df.shape[0], 2)
missing_info[missing_info > 0].sort_values(ascending= False)


# * Above 19 columns have missing values.
# * It can be seen that PoolQC, MiscFeature, Alley, Fence and FireplaceQu have very high percentage of missing value.
# * I'll check all the columns to understand if these are actually missing or these have some meaning. Once it's identified, imputation can be performed:
#     * Imputation using Business Knowledge : If NaN signifies any business logic, then we can impute NaN with that.
#     * Statistical Imputation: Statistical imputation methods can be used to impute missing values after train-test split.

# In[9]:


# Getting column names having missing values
missing_val_cols= missing_info[missing_info > 0].sort_values(ascending= False).index
missing_val_cols


# In[10]:


# Checking unique values in these columns
for col in missing_val_cols:
  print('\nColumn Name:',col)
  print(housing_df[col].value_counts(dropna= False))


# * In Data Dictionary it's mentioned that, NA for below features denote that these features are not present for the house

# <b> # 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'

# So here, we will replace NaN values for above attributes withh 'Not Present'.
# 
# * For rest of the columns we'll check if they have any relation with other columns and if we can use that relation in observed daat to impute these columns:

# <b>LotFrontage, GarageYrBlt, MasVnrArea, MasVnrType, Electrical

# In[11]:


# Replacing NaN with 'Not Present' for below columns
valid_nan_cols= ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
housing_df[valid_nan_cols]= housing_df[valid_nan_cols].fillna('Not Present')


# In[12]:


# Checking percentage of missing values again
missing_info= round(housing_df.isna().sum() * 100/housing_df.shape[0], 2)
missing_info[missing_info > 0].sort_values(ascending= False)


# In[13]:


# Checking if there is any relation between GarageYrBlt and GarageType
housing_df[housing_df.GarageYrBlt.isna()]['GarageType'].value_counts(normalize= True)


# Initially GarageYrBlt and GarageType both had 5.55% missing value. After imputing NaN values of GarageType with 'Not Available', we can see that GarageYrBlt value is NaN for only those observations where GarageType is 'Not Available'. We can conclude that if garage is not available then there will be no 'GarageYrBlt' value for that. So we can safely <b>impute GarageYrBlt NaN values with 0.

# In[14]:


# Imputing missing values of GarageYrBlt column
housing_df['GarageYrBlt']= housing_df['GarageYrBlt'].fillna(0)


# I'll perform statistical imputation for rest of the columns after train-test split: <b>LotFrontage, MasVnrArea, MasVnrType, Electrical

# # Changing data types

# MSSubClass: "identifies the type of dwelling involved in the sale", is a categorical variable, but it's appearing as a numeric variable.

# In[15]:


# Changing data type of MSSubClass
housing_df['MSSubClass']= housing_df['MSSubClass'].astype('object')


# # Exploratory Data Analysis

# There are 81 attributes in the dataset. So, I am running SweetViz AutoEDA to explore and vizualize the data. The I'll manually explore the attributes that have high correlation coefficient with the target variable.

# # AutoEDA using SweetViz

# In[16]:


# Running SweetViz AutoEDA
sv_report= sv.analyze(housing_df)
sv_report.show_notebook()


# In[17]:


# Saving the report as html
sv_report.show_html(r'E:\Internships\Fliprobo\HousingProject\AutoEDA_report.html')


# # Observations from AutoEDA

# <b>Numerical Associations with SalePrice:

# * GrLivArea: 0.71
# * GarageArea: 0.62
# * TotalBsmtSF: 0.61
# * 1stFlrSF: 0.61
# * TotRmsAbvGrd: 0.53
# * YearBuilt: 0.52
# * YearRemodAdd: 0.51
# * MasVnrArea: 0.48
# * BsmtFinSF1: 0.39
# * LotFrontage: 0.35
# * WoodDeckSF: 0.32
# * 2ndFlrSF: 0.32
# * OpenPorchSF: 0.32
# * LotArea: 0.26

# <b>Categorical Associations with SalePrice:

# * OverallQual: 0.83
# * Neighborhood: 0.74
# * GarageCars: 0.70
# * ExterQual: 0.69
# * BsmtQual: 0.68
# * KitchenQual: 0.68
# * FullBath: 0.58
# * GarageFinish: 0.55
# * FireplaceQu: 0.54
# * Foundation: 0.51
# * GarageType: 0.50
# * Fireplaces: 0.48
# * BsmtFinType1: 0.46
# * HeatingQC: 0.44

# # Visualizing numeric variables:

# In[18]:


# Checking distribution of SalePrice
sns.distplot(housing_df['SalePrice'])


# In[19]:


# Plotting numeric variables against SalePrice

numeric_cols= ['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','TotRmsAbvGrd','YearBuilt','YearRemodAdd','MasVnrArea',
'BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','LotArea']

sns.pairplot(housing_df, x_vars=['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','TotRmsAbvGrd'], y_vars='SalePrice', kind= 'reg', plot_kws={'line_kws':{'color':'teal'}})
sns.pairplot(housing_df, x_vars=['YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','LotFrontage'], y_vars='SalePrice', kind= 'reg', plot_kws={'line_kws':{'color':'teal'}})
sns.pairplot(housing_df, x_vars=['WoodDeckSF','2ndFlrSF','OpenPorchSF','LotArea'], y_vars='SalePrice', kind= 'reg', plot_kws={'line_kws':{'color':'teal'}})


# # Visualizing categorical variables:

# In[20]:


# Box plot of catego

cat_cols= ['OverallQual','GarageCars','ExterQual','BsmtQual','KitchenQual','FullBath','GarageFinish','FireplaceQu','Foundation','GarageType','Fireplaces','BsmtFinType1','HeatingQC']

plt.figure(figsize=[18, 40])

for i, col in enumerate(cat_cols, 1):
    plt.subplot(7,2,i)
    title_text= f'Box plot {col} vs cnt'
    x_label= f'{col}'
    fig= sns.boxplot(data= housing_df, x= col, y= 'SalePrice', palette= 'Greens')
    fig.set_title(title_text, fontdict= { 'fontsize': 18, 'color': 'Green'})
    fig.set_xlabel(x_label, fontdict= {'fontsize': 12, 'color': 'Brown'})
plt.show()


# In[21]:


plt.figure(figsize=[17,7])
sns.boxplot(data= housing_df, x= 'Neighborhood', y= 'SalePrice', palette= 'Greens')
plt.show()


# # Inferences

# * SalePrice is right sckewed and other numeic feature: 'GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','TotRmsAbvGrd','YearBuilt','YearRemodAdd','MasVnrArea',
# 
# * <b>'BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','LotArea' have outlier and they all have somewhat linear relation with SalePrice.
# 
# * Median SalePrice is higher for the houses with higher OverallQual rating. Houses with Excellent quality of the material on the exterior have highest price. Price reduces as quality decreases.
# * SalePrice is high for houses having Neighbourhood : Northridge Heights, Stone Brook, Northridge. price is comparatively lower in Iowa DOT and Rail Road, Meadow Village neighbourhood.
# * Median SalePrice is higher for the houses having Excellent Heating quality and median price reduces with Heating quality type and least for the houses having Poor heating quality.
# * Median SalePrice is very high for Good Living Quarters type basement finished area and if the beasement height is more than 100+ inches and least for the houses not having basement.
# * Houses having atleast 1 fireplace, have higher median SalePrice. If Fireplace quality is 'Excellent - Exceptional Masonry Fireplace' then the median SalePrice is the highest.
# * Houses having a garage as part of the house (typically has room above garage) and garage interior 'finish' or 'Rough Finished', have higest median SalePrice. Price is lower for the houses having no garage.
# * Houses with garage in car capacity of 3 have highest median SalePrice.
# * Houses having Poured Contrete foundation has higher SalePrice. Price for houses having Stone and Wood foundations is positive skewed.
# * SalePrice is high for houses with 3 Full bathrooms above grade.

# # Correlation Heatmap

# In[22]:


# Creating correlation heatmap
plt.figure(figsize = (20, 12))
sns.heatmap(housing_df.corr(), annot= True, cmap= 'coolwarm', fmt= '.2f', vmin= -1, vmax= 1)
plt.show()


# Below features have very high correlation coefficients.
# 
# * GrLivArea and TotRmsAbvGrd= .83
# * GarageCars and GarageArea= .88

# In[23]:


# Dropping GarageCars and TotRmsAbvGrd
housing_df.drop(['GarageCars','TotRmsAbvGrd'], axis= 1, inplace= True)
housing_df.shape


# In[24]:


housing_df_org= housing_df.copy()


# # Data Preparation

# Earlier we have already seen that our target variable SalePrice is heavily right skewed. We can perform log transformation to remove the skewness. It will help to boost model performance.

# <b>Transforming the Target variable

# In[25]:


# Distplot of log transformed SalePrice
sns.distplot(np.log(housing_df['SalePrice']))
plt.show()


# It can be seen that after log transformation SalePrice has now near normal distribution.

# In[26]:


# Transforming 'SalePrice'
housing_df['SalePrice_log_trans']= np.log(housing_df['SalePrice'])


# Now, Dropping SalePrice as we have ceate log transformed of it. Also dropping Id column, as it'll not help in predicction.

# # Dropping unnecessary columns

# In[27]:


# Dropping ID Column and SalePrice
housing_df.drop(['SalePrice','Id'], axis=1, inplace= True)
housing_df.shape


# # Train-Test split

# In[28]:


# Train-Test Split
y= housing_df['SalePrice_log_trans']
X= housing_df.drop('SalePrice_log_trans', axis= 1)

X_train, X_test, y_train, y_test= train_test_split(X, y, train_size= .7, random_state= 42)


# In[29]:


# Getting index values of train test dataset
train_index= X_train.index
test_index= X_test.index


# # Statistical imputation of missing values

# Imputing rest of the features in test and train dataset using median (for continuous variables) and mode (for categorical variables) calculated on train dataset.

# In[30]:


# Performing Statistical Imputation for missing values in LotFrontage, MasVnrArea, MasVnrType, Electrical columns

housing_df['LotFrontage'].fillna(X_train['LotFrontage'].median(), inplace= True)
housing_df['LotFrontage'].fillna(X_train['LotFrontage'].median(), inplace= True)

housing_df['MasVnrArea'].fillna(X_train['MasVnrArea'].median(), inplace= True)
housing_df['MasVnrArea'].fillna(X_train['MasVnrArea'].median(), inplace= True)

housing_df['MasVnrType'].fillna(X_train['MasVnrType'].mode(), inplace= True)
housing_df['MasVnrType'].fillna(X_train['MasVnrType'].mode(), inplace= True)

housing_df['Electrical'].fillna(X_train['Electrical'].mode(), inplace= True)
housing_df['Electrical'].fillna(X_train['Electrical'].mode(), inplace= True)


# # Encoding categorical (nominal) features

# In[31]:


# Getting object and numeric type columns
housing_cat= housing_df.select_dtypes(include= 'object')
housing_num= housing_df.select_dtypes(exclude= 'object')
housing_cat.describe()


# In[32]:


# 'Street','Utilities', 'CentralAir' have 2 unique data, so we are encoding with 0 and 1
housing_df['Street']= housing_df.Street.map(lambda x: 1 if x== 'Pave' else 0)
housing_df['Utilities']= housing_df.Utilities.map(lambda x: 1 if x== 'AllPub' else 0)
housing_df['CentralAir']= housing_df.CentralAir.map(lambda x: 1 if x== 'Y' else 0)


# For rest of the categorical (Nominal) columns One Hot Encoding will be used.

# In[33]:


# Performing get_dummies
cat_cols= housing_cat.columns.tolist()
done_encoding= ['Street','Utilities', 'CentralAir']
cat_cols= [col for col in cat_cols if col not in done_encoding]
dummies= pd.get_dummies(housing_df[cat_cols], drop_first=True)


# In[34]:


# Checking all dummies
dummies.head()


# In[35]:


# Concatinating dummies with housing_df dataframe and droping original features
print('housing_df before droping original valiables', housing_df.shape)
print('shape of dummies dataframe', dummies.shape)
housing_df.drop(cat_cols, axis=1, inplace= True)
housing_df= pd.concat([housing_df, dummies], axis= 1)
print('final shape of housing_df', housing_df.shape)


# # Scaling numeric features

# During EDA we have observed few outliers in numeric features. So, using Robust Scaling using median and quantile values instead of Standard Scaling using mean and standard deviation.

# In[36]:


# Re-constructing Train-test data
X_train= housing_df.iloc[train_index, :].drop('SalePrice_log_trans', axis= 1)
y_train= housing_df.iloc[train_index, :]['SalePrice_log_trans']
X_test= housing_df.iloc[test_index, :].drop('SalePrice_log_trans', axis= 1)
y_test= housing_df.iloc[test_index, :]['SalePrice_log_trans']


# In[37]:


# Performing scaling of numeric columns in training and test dataset using RobustScaler
num_cols= housing_num.columns.tolist()
num_cols.remove('SalePrice_log_trans')
scaler= RobustScaler(quantile_range=(2, 98))
scaler.fit(X_train[num_cols])
X_train[num_cols]= scaler.transform(X_train[num_cols])
X_test[num_cols]= scaler.transform(X_test[num_cols])


# In[38]:


# Checking scaled features
X_train[num_cols].head()


# # Variance Thresholding

# During EDA, we have seen that there are few categorical features where only a handful of observations differ from a constant value. Remvoing those categorical features having zero or close to zero variance.

# In[39]:


var_t= VarianceThreshold(threshold= .003)
variance_thresh= var_t.fit(X_train)
col_ind= var_t.get_support()

# Below columns have very low variance
X_train.loc[:, ~col_ind].columns


# In[40]:


# Checking number of apperance of one of the attributes/categorical value in dataset
housing_df_org.Functional.value_counts()


# It can be seen that Functional_Sev or Functional with 'Sev' type has only one observation in entire dataset.

# In[41]:


# Removing above columns from train and test dataset
X_train= X_train.loc[:, col_ind]
X_test= X_test.loc[:, col_ind]


# In[42]:


# Checking shape of final training dataset
X_train.shape


# # Model Building

# <b>Ridge Regression

# In[43]:


# Selecting few values for alpha
range1= [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
range2= list(range(2, 1001))
range1.extend(range2)
params_grid= {'alpha': range1}


# In[44]:


# Applying Ridge and performing GridSearchCV to find optimal value of alpha (lambda)

ridge= Ridge(random_state= 42)
gcv_ridge= GridSearchCV(estimator= ridge, 
                        param_grid= params_grid,
                        cv= 3,
                        scoring= 'neg_mean_absolute_error',
                        return_train_score= True,
                        n_jobs= -1,
                        verbose= 1)      
gcv_ridge.fit(X_train, y_train)


# In[45]:


# Checking best estimator 
gcv_ridge.best_estimator_


# In[46]:


# Checking best MAE
gcv_ridge.best_score_


# Optimal value for alpha is 8.

# In[47]:


# Fitting model using best_estimator_
ridge_model= gcv_ridge.best_estimator_
ridge_model.fit(X_train, y_train)


# In[48]:


# Evaluating on training dataset
y_train_pred= ridge_model.predict(X_train)
print( 'r2 score on training dataset:', r2_score(y_train, y_train_pred))
print( 'MSE on training dataset:', mean_squared_error(y_train, y_train_pred))
print( 'RMSE on training dataset:', (mean_squared_error(y_train, y_train_pred)**.5))
print( 'MAE on training dataset:', mean_absolute_error(y_train, y_train_pred))


# In[49]:


# Evaluating on testing dataset
y_test_pred= ridge_model.predict(X_test)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))


# In[50]:


# Ridge coefficients
ridge_model.coef_


# In[51]:


# Ridge intercept
ridge_model.intercept_


# In[52]:


# Top 10 features with double the value of optimal alpha in Ridge
ridge_coef= pd.Series(ridge_model.coef_, index= X_train.columns)
top_25_ridge=  ridge_coef[abs(ridge_coef).nlargest(25).index]
top_25_ridge


# # Lasso Regression

# In[53]:


# Applying Lasso and performing GridSearchCV to find optimal value of alpha (lambda)

params_grid= {'alpha': range1}
lasso= Lasso(random_state= 42)
lasso_gcv= GridSearchCV(estimator= lasso, 
                        param_grid= params_grid,
                        cv= 3,
                        scoring= 'neg_mean_absolute_error',
                        return_train_score= True,
                        n_jobs= -1,
                        verbose= 1)

lasso_gcv.fit(X_train, y_train)  


# In[54]:


# Checking best estimator 
lasso_gcv.best_estimator_


# In[55]:


# Checking best MAE
lasso_gcv.best_score_


# Optimal value for alpha is .0001. Next I'll try to fine tune this value by running GridSearchCV with some closer values to .0001

# In[56]:


range3= [0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001, .0002, .0003, .0004, .0005, .0006, .0007, .0008, .0009, .001]
params_grid= {'alpha': range3}
lasso_gcv= GridSearchCV(estimator= lasso, 
                        param_grid= params_grid,
                        cv= 3,
                        scoring= 'neg_mean_absolute_error',
                        return_train_score= True,
                        n_jobs= -1,
                        verbose= 1)

lasso_gcv.fit(X_train, y_train) 


# In[57]:


# Checking best estimator 
lasso_gcv.best_estimator_


# So, for Lasso we are getting optimal value of alpha as .0006.

# In[58]:


# Fitting model using best_estimator_
lasso_model= lasso_gcv.best_estimator_
lasso_model.fit(X_train, y_train)


# In[59]:


# Evaluating on training dataset
y_train_pred= lasso_model.predict(X_train)
print( 'r2 score on training dataset:', r2_score(y_train, y_train_pred))
print( 'MSE on training dataset:', mean_squared_error(y_train, y_train_pred))
print( 'RMSE on training dataset:', (mean_squared_error(y_train, y_train_pred)**.5))
print( 'MAE on training dataset:', mean_absolute_error(y_train, y_train_pred))


# In[60]:


# Evaluating on testing dataset
y_test_pred= lasso_model.predict(X_test)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))


# In[61]:


# Checking no. of features in Ridge and Lasso models
lasso_coef= pd.Series(lasso_model.coef_, index= X_train.columns)
selected_features= len(lasso_coef[lasso_coef != 0])
print('Features selected by Lasso:', selected_features)
print('Features present in Ridge:', X_train.shape[1])


# In[62]:


# Lasso intercept
lasso_model.intercept_


# In[63]:


# Top 25 features with coefficients in Lasso model
top25_features_lasso=  lasso_coef[abs(lasso_coef[lasso_coef != 0]).nlargest(25).index]
top25_features_lasso


# # Conclusion

# * Ridge and Lasso both the models have almost same test and train accuracy. So it can be said that there is no overfitting.
# 
# * Lasso and Ridge both have similar r2 score and MAE on test dataset. But Lasso has eliminated 110 features and final no. of features in Lasso Regression model is 116. Where Ridge has all 226 features. So, our Lasso model is simpler than Ridge with having similar r2 score and MAE.
# 
#       * Ridge Regression model on test dataset: r2 score= 0.8912, MAE= 0.0934, RMSE= 0.1357
#       * Lasso Regression model on test dataset: r2 score= 0.8947, MAE= 0.0914, RMSE= 0.1335
# * Considering above points we can choose our Lasso Regression model as our final model.
# 
# * Below are the top 25 features in the Lasso regression model.

# In[64]:


# Ploting top 25 features
plt.figure(figsize= (7, 5))
top25_features_lasso.plot.barh(color= (top25_features_lasso > 0).map({True: 'g', False: 'r'}))
plt.show()


# * Optimal alpha (lambda) value for Ridge Regression model is: 8
# * Optimal alpha (lambda) value for Lasso Regression model is: 0.0006

# # Scenario 1: Doubling the value of optimal alpha

# In[65]:


## Doubling value of optimal alpha in Ridge
ridge2= Ridge(alpha= 16, random_state= 42)
ridge2.fit(X_train, y_train)


# In[66]:


# Top 10 features with double the value of optimal alpha in Ridge
ridge_coef2= pd.Series(ridge2.coef_, index= X_train.columns)
top10_ridge2=  ridge_coef2[abs(ridge_coef2).nlargest(10).index]
top10_ridge2


# In[67]:


## Doubling value of optimal alpha in Lasso
lasso2= Lasso(alpha= .0012, random_state=42)
lasso2.fit(X_train, y_train)


# In[68]:


# Top 10 features with double the value of optimal alpha in Lasso
lasso_coef2= pd.Series(lasso2.coef_, index= X_train.columns)
top10_lasso2=  lasso_coef2[abs(lasso_coef2[lasso_coef2 != 0]).nlargest(10).index]
top10_lasso2


# # Scenario 2: 5 most important predictor variables in the lasso model are not available in the incoming data

# In[69]:


# Checking top 5 features in our lasso model
top25_features_lasso.nlargest()


# As Neighborhood_StoneBr is a dummy variable, we'll drop entire Neighborhood feature.

# In[70]:


# Checking all Neighborhood dummy variables
cols_to_drop= X_train.columns[X_train.columns.str.startswith('Neighborhood')].tolist()
cols_to_drop.extend(['GrLivArea','OverallQual','OverallCond','GarageArea'])
cols_to_drop


# In[71]:


# Droping above features from X_train and X_test
X_train= X_train.drop(cols_to_drop, axis= 1)
X_test= X_test.drop(cols_to_drop, axis= 1)
X_train.shape, X_test.shape


# In[72]:


# Building Lasso model with these features
lasso3= Lasso(alpha= .0006, random_state= 42)
lasso3.fit(X_train, y_train)


# In[73]:


# Top 5 features after droping top 5 features of Previous Lasso model
lasso_coef3= pd.Series(lasso3.coef_, index= X_train.columns)
top5_lasso3=  lasso_coef3[abs(lasso_coef3[lasso_coef3 != 0]).nlargest().index]
top5_lasso3


# In[ ]:




