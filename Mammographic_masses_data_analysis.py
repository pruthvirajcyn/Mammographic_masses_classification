#!/usr/bin/env python
# coding: utf-8

# # Mammogramin mass data analysis
# 
# ## Predict whether a mammogram mass is benign or malignant
# 
# We'll be using the "mammographic masses" public dataset from the UCI repository (source: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)
# 
# This data contains 961 instances of masses detected in mammograms, and contains the following attributes:
# 
# 
#    1. BI-RADS assessment: 1 to 5 (ordinal)  
#    2. Age: patient's age in years (integer)
#    3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
#    4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
#    5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
#    6. Severity: benign=0 or malignant=1 (binominal)
#    
# BI-RADS is an assesment of how confident the severity classification is; it is not a "predictive" attribute and so we will discard it. The age, shape, margin, and density attributes are the features that we will build our model with, and "severity" is the classification we will attempt to predict based on those attributes.
# 
# Although "shape" and "margin" are nominal data types, which sklearn typically doesn't deal with well, they are close enough to ordinal that we shouldn't just discard them. The "shape" for example is ordered increasingly from round to irregular.
# 
# A lot of unnecessary anguish and surgery arises from false positives arising from mammogram results. If we can build a better way to interpret them through supervised machine learning, it could improve a lot of lives.
# 
# Let's apply several different supervised machine learning techniques to this data set, and see which one yields the highest accuracy as measured with K-Fold cross validation (K=10). Apply:
# 
# * Decision tree
# * Random forest
# * KNN
# * Naive Bayes
# * SVM
# * Logistic Regression
# * Neural network using Keras.
# 
# 

# ## Prepare our data
# 
# We start by importing the mammographic_masses.data.txt file into a Pandas dataframe using read_csv and take a look at it.

# In[1]:


import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
#import seaborn as sn


# We make sure to convert missing data indicated by a ? into NaN, and to add the appropriate column names (BI_RADS, age, shape, margin, density, and severity):

# In[4]:


Attribute_names=['BI-rads','Age','Shape','Margin','Density','Severity']
mammographic_masses=pd.read_csv("./mammographic_masses.data.txt", na_values=['?'], names= Attribute_names)
mammographic_masses


#  We evaluate whether the data needs cleaning by using describe() on the dataframe.

# In[5]:


mammographic_masses.describe()


# There are quite a few missing values in the data set. Before we just drop every row that's missing data, let's make sure we don't bias our data in doing so. Does there appear to be any sort of correlation to what sort of data has missing fields? If there were, we'd have to try and go back and fill that data in.

# In[2]:


scatter_matrix(mammographic_masses)
plt.show()


# If the missing data seems randomly distributed, we can go ahead and drop rows with missing data.by using dropna().

# In[7]:


data=mammographic_masses.dropna(axis=0)


# Next we'll need to convert the Pandas dataframes into numpy arrays that can be used by scikit_learn. Create an array that extracts only the feature data we want to work with (age, shape, margin, and density) and another array that contains the classes (severity). We'll also need an array of the feature name labels.

# In[8]:


X_array=data[['Age','Shape','Margin','Density']].values
Y_array=data['Severity'].values
X_array
X_features=['Age','Shape','Margin','Density']


# Some of our models require the input data to be normalized, so we normalize the attribute data using preprocessing.StandardScaler().
# SVM also require the input data to be normalized first.
# 

# In[1]:


from sklearn import preprocessing
Scalar=preprocessing.StandardScaler()
Scaled_X=Scalar.fit_transform(X_array)
Scaled_X


# ## Decision Trees
# 
# Before moving to K-Fold cross validation and random forests, start by creating a single train/test split of our data. Set aside 75% for training, and 25% for testing.

# In[6]:


from sklearn.model_selection import train_test_split
from sklearn import tree
(Training_inputs,Test_inputs,Training_output,Test_output)=train_test_split(Scaled_X,Y_array,train_size=0.75,random_state=31)


# Now create a DecisionTreeClassifier and fit it to our training data.

# In[7]:


from sklearn.tree import DecisionTreeClassifier
DecTree=DecisionTreeClassifier(max_depth=5,random_state=31)
DesResult=DecTree.fit(Training_inputs,Training_output)


# Display the resulting decision tree.

# In[8]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus

dot_data = StringIO()  
tree.export_graphviz(DesResult, out_file=dot_data,  
                         feature_names=X_features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  


# Measure the accuracy of the resulting decision tree model using your test data.

# In[9]:


DesResult.score(Test_inputs,Test_output)


# 

# In[10]:


from sklearn.model_selection import cross_val_score

DecTree= DecisionTreeClassifier(random_state=31)

scores=cross_val_score(DecTree,Scaled_X,Y_array,cv=10)
scores.mean()


# Now try a RandomForestClassifier instead. Does it perform better?

# In[11]:


from sklearn.ensemble import RandomForestClassifier
a=RandomForestClassifier(n_estimators=10)
scores=cross_val_score(a, Scaled_X,Y_array, cv=10)
scores.mean()


# ## SVM
# 
# Next try using svm.SVC with a linear kernel. How does it compare to the decision tree?

# In[15]:


from sklearn import svm
SVM=svm.SVC(kernel='linear',C=1,gamma='scale')


# In[16]:


scores=cross_val_score(SVM,Scaled_X,Y_array,cv=10)
scores.mean()


# ## KNN
# Now we would like to try K nearest neighbours. We start with a K of 10. K is an example of a hyperparameter - a parameter on the model itself which may need to be tuned for best results on your particular data set.

# In[17]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
scores=cross_val_score(knn,Scaled_X,Y_array,cv=10)
scores.mean()


# Choosing K is tricky, so we can't discard KNN until we've tried different values of K. Write a for loop to run KNN with K values ranging from 1 to 50 and see if K makes a substantial difference. Make a note of the best performance you could get out of KNN.

# In[18]:


diffkscores=[]
for k in range(1,50):
    knn=KNeighborsClassifier(n_neighbors=k)
    diffk=cross_val_score(knn,Scaled_X,Y_array,cv=10)
    diffkscores.append(diffk.mean())
max(diffkscores)


# ## Naive Bayes
# 
# Now we try naive_bayes.MultinomialNB.

# In[19]:


from sklearn.naive_bayes import MultinomialNB

scaler = preprocessing.MinMaxScaler()
all_features_minmax = scaler.fit_transform(X_array)

nb = MultinomialNB()
scores = cross_val_score(nb, all_features_minmax, Y_array, cv=10)
scores.mean()


# ## Revisiting SVM
# 
# svm.SVC may perform differently with different kernels. The choice of kernel is an example of a "hyperparamter." Try the rbf, sigmoid, and poly kernels and see what the best-performing kernel is.

# In[20]:


SVM=svm.SVC(kernel='poly',C=1,gamma='scale')
scores=cross_val_score(SVM,Scaled_X,Y_array,cv=10)
scores.mean()


# In[21]:


SVM=svm.SVC(kernel='sigmoid',C=1,gamma='scale')
scores=cross_val_score(SVM,Scaled_X,Y_array,cv=10)
scores.mean()


# In[22]:


SVM=svm.SVC(kernel='rbf',C=1,gamma='scale')
scores=cross_val_score(SVM,Scaled_X,Y_array,cv=10)
scores.mean()


# ## Logistic Regression
# 
# We've tried all these fancy techniques, but fundamentally this is just a binary classification problem. Try Logisitic Regression, which is a simple way to tackling this sort of thing.

# In[23]:


from sklearn.linear_model import LogisticRegression

LR=LogisticRegression(solver='lbfgs')
scores=cross_val_score(LR,Scaled_X, Y_array,cv=10)
scores.mean()


# ## Neural Networks
# 
# Let's see if an Deep neural network can do even better. 

# In[11]:


from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def NNmodel():
    model=Sequential()
    model.add(Dense(32, input_dim=4,kernel_initializer='normal',activation='relu'))
    model.add(Dense(16,kernel_initializer='normal',activation='relu'))
    model.add(Dense(4,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

estimator=KerasClassifier(build_fn=NNmodel, epochs=1, verbose=0)

nn_scores=cross_val_score(estimator,Scaled_X,Y_array,cv=10)
print(nn_scores.mean())


# In[ ]:





# In[ ]:





# In[ ]:





# ## Do we have a winner?
# 

# We can clearly see that ogistic regression performed the best. Even its a very simple classification machine leanring algorithm, the best resuts we obtained.

# In[ ]:




