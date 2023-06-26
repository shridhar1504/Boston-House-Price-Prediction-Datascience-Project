#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction - Boston Dataset

# ***
# _**Importing the required libraries, packages & dataset**_

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import ydata_profiling as pf
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')


# ## Dataset Cleaning:
# _**Making the loaded Boston dataset as an usable Dataset for prediction**_

# In[2]:


boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target


# _**Changing The Default Working Directory Path**_

# In[3]:


os.chdir('C:\\Users\\Shridhar\\OneDrive\\Desktop\\Boston House Price Prediction')


# _**Exporting the dataset to a csv file for readability**_

# In[4]:


data.to_csv('Boston House Price Data.csv',index = False)


# _**Reading the Dataset using Pandas Command**_

# In[5]:


df = pd.read_csv('Boston House Price Data.csv')


# ## Data Visualization:

# _**Automated Exploratory Data Analysis (EDA) with ydata_profiling(pandas_profiling)**_

# In[6]:


pf.ProfileReport(df)


# _**Getting the Correlation Values from all the numeric columns from the dataset and checking for correlation using Seaborn Heatmap & saving the PNG File**_

# In[7]:


plt.rcParams['figure.figsize']=(20,15)
sns.heatmap(df.corr(),annot = True, cmap = 'Greens',square = True,cbar = True)
plt.title('Correlation Heat Map')
plt.savefig('Correlation Heat Map.png')
plt.show()


# _**Assigning the dependent and independent variable**_

# In[8]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# ## Model Fitting:

# _**Defining the Function for the ML algorithms using GridSearchCV Algorithm and splitting the dependent variable & independent variable into training and test dataset and Predicting the Dependent Variable by fitting the given model and create the pickle file of the model with the given Algo_name. Further getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset.**_

# In[9]:


def FitModel(x,y,algo_name,algorithm,GridSearchParams,cv):
    np.random.seed(10)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 99)
    grid = GridSearchCV(estimator = algorithm,param_grid = GridSearchParams,cv = cv,
                     scoring = 'r2',verbose = 0, n_jobs = -1)
    grid_result = grid.fit(x_train,y_train)
    best_params = grid_result.best_params_
    pred = grid_result.predict(x_test)
    pickle.dump(grid_result,open(algo_name,'wb'))
    print('Algorithm Name:\t',algo_name)
    print('Best Params:',best_params)
    print('R2 Score : {}%'.format(100* r2_score(y_test,pred)))             


# _**Running the function with empty parameters since the Linear Regression model doesn't need any special parameters and fitting the Linear Regression Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name Linear Regression.**_

# In[10]:


param = {}
FitModel(x,y,'Linear Regression',LinearRegression(),param,cv=10)


# _**Running the function with empty parameters since the Lasso model doesn't need any special parameters and fitting the Lasso Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name Lasso.**_

# In[11]:


param = {}
FitModel(x,y,'Lasso',Lasso(),param,cv=10)


# _**Running the function with empty parameters since the Ridge model doesn't need any special parameters and fitting the Ridge Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name Ridge.**_

# In[12]:


param={}
FitModel(x,y,'Ridge',Ridge(),param,cv=10)


# _**Running the function with some appropriate parameters and fitting the Decision Tree Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name Decision Tree.**_

# In[13]:


param = { "max_features":['auto','sqrt'],
          "max_depth":[int(x) for x in np.linspace(6, 45, num = 5)],
          "min_samples_leaf":[1,2,5,10],
          "min_samples_split":[2, 5, 10, 15, 100],
          "min_weight_fraction_leaf":[0.0,0.1,0.2,0.3,0.4,0.5]}

FitModel(x,y,'Decision Tree',DecisionTreeRegressor(),param,cv=10)


# _**Running the function with some appropriate parameters and fitting the Random Forest Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name Random Forest.**_

# In[14]:


param = {'n_estimators':[500,600,800,1000],
         "criterion":["squared_error", "absolute_error", "poisson"]}
FitModel(x,y,'Random Forest',RandomForestRegressor(),param,cv=10)


# _**Running the function with some appropriate parameters and fitting the Extra Trees Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name Extra Trees.**_

# In[15]:


param = {'n_estimators':[500,600,800,1000],
        'max_features':['auto','sqrt']}
FitModel(x,y,'Extra Trees',ExtraTreesRegressor(),param,cv=10)


# _**Running the function with some appropriate parameters and fitting the XGBoost Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name XGBoost.**_

# In[16]:


param = {"n_estimators":[111,222,333,444]}
FitModel(x,y,'XGBoost',XGBRegressor(),param,cv=10)


# _**Running the function with empty parameters since the CatBoost Regressor model doesn't need any special parameters and fitting the CatBoost Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name CatBoost.**_

# In[17]:


param = {}
FitModel(x,y,'CatBoost',CatBoostRegressor(),param,cv=10)


# _**Running the function with empty parameters since the LightGBM Regressor model doesn't need any special parameters and fitting the LightGBM Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name LightGBM.**_

# In[18]:


param = {}
FitModel(x,y,'LightGBM',LGBMRegressor(),param,cv=10)


# _**Loading the pickle file with the algorithm which gives highest r2 score percentage**_

# In[19]:


model=pickle.load(open('CatBoost','rb'))


# _**Predicting the dependent variable using the loaded pickle file and getting the Accuracy Score in percentage format between the predicted values and dependent variable**_

# In[20]:


pred1=model.predict(x)
print('R2 Score :{}%'.format(100* r2_score(y,pred1)))


# _**Making the Predicted value as a new dataframe and concating it with the original data, so that we can able to compare the differences between Predicted price and Original Price.**_

# In[21]:


prediction = pd.DataFrame(pred1,columns=['Predicted Price(Approx.)'])
pred_df = pd.concat([df,prediction],axis=1)


# _**Exporting the Data With Prediction of House Price to a csv file**_

# In[22]:


pred_df.to_csv('Boston Predicted House Price.csv', index = False)


# _**Plotting the line graph to represent the Accuracy between Predicted Price and Original Price and saving the PNG file**_

# In[23]:


plt.plot(y,pred1)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.savefig('Actual Price vs Predicted Price.png')
plt.show()

