
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')

from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Imputer , Normalizer

def heatMap(df):
    #Create Correlation df
    corr = df.corr()
    #Plot figsize
    fig, ax = plt.subplots(figsize=(10, 10))
    #Generate Color Map, red & blue
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    #Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    #Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns);
    #Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns)
    #show plot
    plt.show()


# In[2]:


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
combined = [train_data, test_data]


# In[3]:


train_data.columns.values


# In[4]:


train_data.dtypes


# In[5]:


#dropping Name , Ticket and PassengerId
train_data = train_data.drop(['Name'], axis=1)
train_data = train_data.drop(['Ticket'], axis=1)
train_data = train_data.drop(['PassengerId'], axis=1)

test_data = test_data.drop(['Name'], axis=1)
test_data = test_data.drop(['Ticket'], axis=1)
combined = [train_data, test_data]


# We see Cabin, Embarked and Sex are object values, we want to convert these to numeric values

# In[6]:


train_data.info()


# Cabin, Age and Embarked has missing values, we will need to handle this. Let's continue analyzing the dataset.

# In[7]:


train_data.describe(include='all')


# just by looking at the above data we can find meaningfull info. 
# 
# Around 38 % survived.
# the mean age was 30 and there were mostly males on the Titanic. 

# In[8]:


##fill in missing values with 'S' (the most common embarked value)
for dataset in combined:
    dataset[ 'Embarked' ] = dataset['Embarked'].fillna('S')


# In[9]:


combined[0].info()


# convert all object values to numerical

# In[10]:


for dataset in combined:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    dataset['Sex'] = dataset['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
    dataset['Cabin'] = pd.get_dummies( dataset.Cabin)


# In[11]:


combined[0].info()


# In[12]:


##filling in missing values with mean
for dataset in combined:
    dataset[ 'Age' ] = dataset.Age.fillna( dataset.Age.mean() )
    dataset[ 'Fare' ] = dataset.Fare.fillna( dataset.Fare.mean() )
    dataset[ 'Cabin' ] = dataset.Cabin.fillna( dataset.Cabin.mean() )


# In[13]:


combined[0].info()


# Now we have no missing values

# In[14]:


heatMap(combined[0])


# When we look at the correlation of the variables, we see that Pclass, Sex and Fare has a strong relation to Survived.

# In[15]:


train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()


# In[16]:


train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# In[17]:


train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()


# In[18]:


train_data['AgeBand'] = pd.cut(train_data['Age'], 5)
train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[19]:


train_data.head()


# In[20]:


for dataset in combined:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_data.head()


# In[21]:


train_data = train_data.drop(['AgeBand'], axis=1)
combine = [train_data, test_data]
test_data.head()


# In[22]:


train_data['Alone'] = 0
train_data.loc[(train_data['SibSp'] > 0) | (train_data['Parch'] > 0), 'Alone'] = 1

test_data['Alone'] = 0
test_data.loc[(test_data['SibSp'] > 0) | (test_data['Parch'] > 0), 'Alone'] = 1


# In[23]:


train_data[['Alone', 'Survived']].groupby(['Alone'], as_index=False).mean()


# In[24]:


train_data = train_data.drop(['Parch', 'SibSp'], axis=1)
test_data = test_data.drop(['Parch', 'SibSp'], axis=1)
combine = [train_data, test_data]


# In[25]:


train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)
train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[26]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_data = train_data.drop(['FareBand'], axis=1)
combine = [train_data, test_data]


# In[27]:


heatMap(train_data)


# In[28]:


X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]
X_test  = test_data.drop("PassengerId", axis=1).copy()


# In[29]:


# Random Forest

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)
Y_prediction = model.predict(X_test)
model.score(X_train, Y_train)
score = round(model.score(X_train, Y_train) * 100, 2)
score


# In[30]:


submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_prediction
    })


# In[31]:


#submission.to_csv('submission.csv', index=False)

