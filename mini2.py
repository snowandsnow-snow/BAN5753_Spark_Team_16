#!/usr/bin/env python
# coding: utf-8

# # Mini Project 2

# In[1]:


import findspark
findspark.init()


# In[2]:


from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import * 
import pyspark.sql.functions as F
from pyspark.sql.functions import col, asc,desc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pyspark.mllib
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline
from sklearn.metrics import confusion_matrix

spark=SparkSession.builder \
.master ("local[*]")\
.appName("part3")\
.getOrCreate()


# In[3]:


from pyspark.sql.functions import col
from pyspark.sql.functions import log


# In[4]:


sc=spark.sparkContext
sqlContext=SQLContext(sc)


# In[5]:


import os
os.getcwd()


# In[6]:


from platform import python_version

print(python_version())


# In[7]:


sc.version #spark version


# ## Read File

# In[8]:


df=spark.read \
 .option("header","True")\
 .option("inferSchema","True")\
 .option("sep",",")\
 .csv("C:/Users/Xue/OneDrive - Oklahoma A and M System/BAN 5753/Mini project/mini 2/XYZ_Bank_Deposit_Data_Classification.csv")
print("There are",df.count(),"rows",len(df.columns),
      "columns" ,"in the data.") 


# # 1. Exploratory Data Analysis (EDA)

# ## 1.1 Data Types of Columns

# In[9]:


df.printSchema()


# ## 1.2 Statistics

# In[10]:


numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
df.select(numeric_features).describe().toPandas().transpose()


# ## 1.3 Check Null values
# * there is no null values in the dataset

# In[11]:


from pyspark.sql.functions import isnan, when, count, col
df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).toPandas().head()


# ## 1.5 Target Variable Distribution
# * From target, we can see the data is unbalanced

# In[12]:


df.groupby("y").count().show()


# ## 1.5 Univariate patterns - Distribution of Features of Numerical Variables
# View Univariate Distributions

# In[14]:


from matplotlib import cm
fig = plt.figure(figsize=(25,15)) ## Plot Size 
st = fig.suptitle("Distribution of Features", fontsize=50,
                  verticalalignment='center') # Plot Main Title 

for col,num in zip(df.toPandas().describe().columns, range(1,12)):
    ax = fig.add_subplot(3,4,num)
    ax.hist(df.toPandas()[col])
    plt.style.use('dark_background') 
    plt.grid(False)
    plt.xticks(rotation=45,fontsize=20)
    plt.yticks(fontsize=15)
    plt.title(col.upper(),fontsize=20)
plt.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85,hspace = 0.4)
plt.show()


# * Except 'AGE' is near normal distributed, all other variables are not normal distributed.
# * These variables will be grouped into two categorices according to their distribution:
# 1. PDAYS
# 2. EURIBOR3M
# * Others will be normalized using 'StandardScaler'.
# 
# 

# ## 1.6 Pearson Correlation
# Identify correlations beteween variables

# In[15]:


numeric_features = [t[0] for t in df.dtypes if t[1] != 'string']
numeric_features_df=df.select(numeric_features)
numeric_features_df.toPandas().head()


# In[16]:


col_names =numeric_features_df.columns
features = numeric_features_df.rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = col_names, col_names

corr_df


# #### Below are the proposed guidelines for the Pearson coefficient correlation interpretation
# ![image.png](attachment:image.png)
# #### Accoring to this guidline,collinarity problem exist in these pairs:
# * Emp_var_rate VS. cons_price_idx
# * Emp_var_rate VS.euribor3m
# * Emp_var_rate VS. nr_employed
# * euribor3m VS. cons_price_idx
# * euribor3m VS. nr_employed
# * Pdays VS. previous
# ####  Deal with these two variables to solve collinearity problem:
# * Emp_var_rate: remove it in modeling building
# * euribor3m: change it to categorical variabe including two groups
# * Pdays : change it to categorical variabe including two groups 

# ## 1.7 Univariate patterns - Distribution of Features of Categorical Variables

# In[17]:


dtypes = df.dtypes
cat_columns = [col for col, dtype in dtypes if dtype == 'string']


# In[29]:


### We can plor distribution for each categorical Variable

from matplotlib import cm
fig = plt.figure(figsize=(25,15)) ## Plot Size 
st = fig.suptitle("Distribution of Features", fontsize=50,
                  verticalalignment='center') # Plot Main Title 

for col,num in zip(cat_columns, range(1,12)):
    a = df.toPandas()[col].value_counts()
    x = list(a.index)
    y = list(a)
    ax = fig.add_subplot(3,4,num)   
    width = 0.75 # the width of the bars 
    ind = np.arange(len(y))  # the x locations for the groups
    ax.barh(ind, y, width)
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(x, minor=False)
    for i, v in enumerate(y):
        ax.text(v + .25, i + .25, str(v), fontweight='bold') #add value labels into bar
    plt.title(col,fontsize=20)
    plt.xlabel('Count',fontsize=20)
    plt.ylabel(col,fontsize=20)
    plt.xticks(rotation=45,fontsize=10)
    plt.yticks(fontsize=15)
plt.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85,hspace = 0.4)
plt.show()


# In[30]:


# We can also check out counts of categorical variables
df.groupBy("job").count().orderBy("count", ascending=False).show(truncate=False)
df.groupBy("marital").count().orderBy("count", ascending=False).show(truncate=False)
df.groupBy("education").count().orderBy("count", ascending=False).show(truncate=False)
df.groupBy("default").count().orderBy("count", ascending=False).show(truncate=False)
df.groupBy("housing").count().orderBy("count", ascending=False).show(truncate=False)
df.groupBy("loan").count().orderBy("count", ascending=False).show(truncate=False)
df.groupBy("contact").count().orderBy("count", ascending=False).show(truncate=False)
df.groupBy("month").count().orderBy("count", ascending=False).show(truncate=False)
df.groupBy("day_of_week").count().orderBy("count", ascending=False).show(truncate=False)
df.groupBy("poutcome").count().orderBy("count", ascending=False).show(truncate=False)
df.groupBy("y").count().orderBy("count", ascending=False).show(truncate=False)
     


# * For some categorical variables which have too many categories but are very unbalanced, they will be grouped into fewer categories:
# 1. Education: college VS. non-college
# 2. Month :Special months (Mar. & Sep. & Otc. & Dec) VS. others
# 

# In[31]:


df.toPandas()


# In[32]:


df.printSchema()


# ## 1.8 Bivariate analysis of target versus Categorical input variables

# In[33]:


df1 = df.toPandas()


# In[34]:


dtypes = df.dtypes
cat_columns = [col for col, dtype in dtypes if dtype == 'string']


# In[35]:


from matplotlib import cm
# fig = plt.figure(figsize=(25,15)) ## Plot Size 
# st = fig.suptitle("Distribution of Features", fontsize=50,
#                   verticalalignment='center') # Plot Main Title 

for col,num in zip(cat_columns, range(1,15)):
#     #print(col)
#     a = df.toPandas()[col].value_counts()
#     x = list(a.index)
#     y = list(a)
    
    width = 0.75 # the width of the bars 
    ind = np.arange(len(y))  # the x locations for the groups
    #ax= plt.subplot(8,2,num) 
    pd.crosstab(df1[col], df1['y']).plot(kind='bar', stacked=False)
      
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(x, minor=False)
#     for i, v in enumerate(y):
#         ax.text(v + .25, i + .25, str(v), fontweight='bold') #add value labels into bar
    plt.title(col,fontsize=20)
    plt.xlabel(col,fontsize=20)
    plt.ylabel("y",fontsize=20)
    plt.xticks(rotation=45,fontsize=10)
    plt.yticks(fontsize=15)
    filename1 = 'C:/Users/Xue/OneDrive - Oklahoma A and M System/BAN 5753/Mini project/mini 2/Figures/'+ str(col) + '.jpg'
    plt.savefig(filename1,bbox_inches='tight')
plt.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85,hspace = 0.4)
plt.show()


# According to the figures above, we can see for some varaibles, the yes rate of subscribe is very unbalanced for each categories of a variable. So next we will calculate Yes Rate in each group to determine how to group some variables. 

# In[36]:


count_y_by_maritalGroup = df.groupBy('marital').agg(
    F.sum(F.when(df['y'] == 'yes', 1).otherwise(0)).alias('count_y_yes'),
    F.sum(F.when(df['y'] == 'no', 1).otherwise(0)).alias('count_y_no')
)

count_y_by_maritalGroup2 = count_y_by_maritalGroup.withColumn('ratio', count_y_by_maritalGroup['count_y_yes'] / (count_y_by_maritalGroup['count_y_yes'] + count_y_by_maritalGroup['count_y_no']))
count_y_by_maritalGroup2 = count_y_by_maritalGroup2.withColumn('yes_rate', F.round(count_y_by_maritalGroup2['ratio'], 2))

count_y_by_maritalGroup2.select('marital','count_y_yes','count_y_no','yes_rate').show()


# In[37]:


count_y_by_educationGroup = df.groupBy('education').agg(
    F.sum(F.when(df['y'] == 'yes', 1).otherwise(0)).alias('count_y_yes'),
    F.sum(F.when(df['y'] == 'no', 1).otherwise(0)).alias('count_y_no')
)

count_y_by_educationGroup2 = count_y_by_educationGroup.withColumn('ratio', count_y_by_educationGroup['count_y_yes'] / (count_y_by_educationGroup['count_y_yes'] + count_y_by_educationGroup['count_y_no']))
count_y_by_educationGroup2 = count_y_by_educationGroup2.withColumn('yes_rate', F.round(count_y_by_educationGroup2['ratio'], 2))

count_y_by_educationGroup2.select('education','count_y_yes','count_y_no','yes_rate').show()


# In[38]:


count_y_by_monthGroup = df.groupBy('month').agg(
    F.sum(F.when(df['y'] == 'yes', 1).otherwise(0)).alias('count_y_yes'),
    F.sum(F.when(df['y'] == 'no', 1).otherwise(0)).alias('count_y_no')
)

count_y_by_monthGroup2 = count_y_by_monthGroup.withColumn('ratio', count_y_by_monthGroup['count_y_yes'] / (count_y_by_monthGroup['count_y_yes'] + count_y_by_monthGroup['count_y_no']))
count_y_by_monthGroup2 = count_y_by_monthGroup2.withColumn('yes_rate', F.round(count_y_by_monthGroup2['ratio'], 2))

count_y_by_monthGroup2.select('month','count_y_yes','count_y_no','yes_rate').show()


# ## 1.9 Bivariate analysis of target versus numerical input variables

# In[46]:


### Histogram for y=yes/no for each numerical variable
from matplotlib import cm
df1 = df.toPandas()['y']=='yes'
fig = plt.figure(figsize=(25,15)) ## Plot Size 
st = fig.suptitle("Distribution of Features", fontsize=50,
                  verticalalignment='center') # Plot Main Title 

for col,num in zip(df.toPandas().describe().columns, range(1,12)):
    ax = fig.add_subplot(3,4,num)
    ax.hist(df.filter(df.y == 'no').toPandas()[col],label='y=no')
    ax.hist(df.filter(df.y == 'yes').toPandas()[col],label='y=yes')
    
    plt.style.use('dark_background') 
    plt.grid(False)
    plt.xticks(rotation=45,fontsize=20)
    plt.yticks(fontsize=15)
    plt.title(col.upper(),fontsize=20)
    plt.legend(loc='upper right')
plt.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85,hspace = 0.4)
plt.show()


# ## 1.10 Prescriptive Recommendations
# Based on this analysis, we have seen a higher subscribe rate among
# 
# * Highly educated
# * Single people
# * Special months (Mar. & Sep. & Otc. & Dec)
# 
# To capitalize on that, we should focus advertising on that demographic with targeted advertising in special months (Mar. & Sep. & Otc. & Dec), highly educated and single people!

# ## 1.10 Group some variables including both numerical and categorical into some groups

# ### a ) Create a new column "pdays_udf" and drop "pdays"

# In[24]:


from pyspark.sql.functions import udf
y_udf = udf(lambda y: 1 if y==999 else 0, StringType())

df=df.withColumn("pdays_udf", y_udf('pdays')).drop("pdays")


# In[25]:


df.groupBy("pdays_udf").count().orderBy("count", ascending=False).show(truncate=False)


# ### b ) Create a new column "euribor3m_udf" and drop "euribor3m"

# In[26]:


from pyspark.sql.functions import udf
y_udf = udf(lambda y: 1 if y>3 else 0, StringType())

df=df.withColumn("euribor3m_udf", y_udf('euribor3m')).drop("euribor3m")


# In[27]:


df.groupBy("euribor3m_udf").count().orderBy("count", ascending=False).show(truncate=False)


# ### c ) Create a new column "education_udf" and drop "education"

# In[28]:


from pyspark.sql.functions import udf
y_udf = udf(lambda y: 1 if y=="university_degree" else 0, StringType())

df=df.withColumn("education_udf", y_udf('education')).drop("education")


# In[29]:


df.groupBy("education_udf").count().orderBy("count", ascending=False).show(truncate=False)


# ### d) Create a new column "month_udf" and drop "month"

# In[30]:


from pyspark.sql.functions import udf
y_udf = udf(lambda y: 1 if (y=="mar") or (y=="sep") or (y=="otc")or (y=="dec") else 0, StringType())

df=df.withColumn("month_udf", y_udf('month')).drop("month")


# In[31]:


df.groupBy("month_udf").count().orderBy("count", ascending=False).show(truncate=False)


# # 2. Prepare Data for Machine Learning

# ## 2.1 StringIndexer

# In[32]:


df2=df


# In[33]:


Indexers = [StringIndexer(inputCol="job", outputCol="jobIndex"), 
            StringIndexer(inputCol="default", outputCol="defaultIndex"), 
            StringIndexer(inputCol="housing", outputCol="housingIndex"),
            StringIndexer(inputCol="loan", outputCol="loanIndex"),
            StringIndexer(inputCol="contact", outputCol="contactIndex"),
            StringIndexer(inputCol="day_of_week", outputCol="day_of_weekIndex"),
            StringIndexer(inputCol="poutcome", outputCol="poutcomeIndex"),                
            StringIndexer(inputCol="pdays_udf", outputCol="pdays_udfIndex"),
            StringIndexer(inputCol="euribor3m_udf", outputCol="euribor3m_udfIndex"),
            StringIndexer(inputCol="education_udf", outputCol="education_udfIndex"),
            StringIndexer(inputCol="marital", outputCol="marital_Index"),
            StringIndexer(inputCol="month_udf", outputCol="month_udfIndex")   
            ]

pipeline = Pipeline(stages=Indexers)
Index_df = pipeline.fit(df2).transform(df2)


# ## 2.2 OneHotEncoderEstimator

# In[34]:


encoder = [OneHotEncoder(inputCol="jobIndex", outputCol="jobencoded"), 
           OneHotEncoder(inputCol="defaultIndex", outputCol="defaultencoded"), 
           OneHotEncoder(inputCol="housingIndex", outputCol="housingencoded"),
           OneHotEncoder(inputCol="loanIndex", outputCol="loanencoded"),
           OneHotEncoder(inputCol="contactIndex", outputCol="contactencoded"),
           OneHotEncoder(inputCol="day_of_weekIndex", outputCol="day_of_weekencoded"),
           OneHotEncoder(inputCol="poutcomeIndex", outputCol="poutcomeencoded"),   
           OneHotEncoder(inputCol="pdays_udfIndex", outputCol="pdays_udfencoded"),
           OneHotEncoder(inputCol="euribor3m_udfIndex", outputCol="euribor3m_udfencoded"),
           OneHotEncoder(inputCol="education_udfIndex", outputCol="education_udfencoded"),
           OneHotEncoder(inputCol="marital_Index", outputCol="marital_encoded"),
           OneHotEncoder(inputCol="month_udfIndex", outputCol="month_encoded")
           ]

pipeline = Pipeline(stages=encoder)
encoder_df = pipeline.fit(Index_df).transform(Index_df)


# ## 2.3 VectorAssembler

# In[35]:


import pandas as pd
pd.set_option('display.max_colwidth', 80)
pd.set_option("display.max_columns", 12)


# In[36]:


assembler = VectorAssembler()\
         .setInputCols (['age',
                         'duration',
                         'campaign',
                         'previous',
                         'jobencoded',
                         'cons_price_idx',
                         'cons_conf_idx',
                         'nr_employed',
                         'duration',
                         'defaultencoded',
                         'housingencoded',
                         'loanencoded',
                         'contactencoded',
                         'day_of_weekencoded',
                         'poutcomeencoded',
                         'pdays_udfencoded',
                         'euribor3m_udfencoded',
                         'education_udfencoded',
                         'marital_encoded',
                         'month_encoded']).setOutputCol ("vectorized_features")
        
# In case of missing you can skip the invalid ones
assembler_df=assembler.setHandleInvalid("skip").transform(encoder_df)
assembler_df.toPandas().head()


# ## 2.4 LabelIndexer

# In[37]:


label_indexer = StringIndexer()\
         .setInputCol ("y")\
         .setOutputCol ("label")

label_indexer_model=label_indexer.fit(assembler_df)
label_indexer_df=label_indexer_model.transform(assembler_df)


# In[38]:


label_indexer_df.toPandas().head(5)


# # 2.5 StandardScaler

# In[39]:


scaler = StandardScaler()\
         .setInputCol ("vectorized_features")\
         .setOutputCol ("features")
        
scaler_model=scaler.fit(label_indexer_df)
scaler_df=scaler_model.transform(label_indexer_df)
pd.set_option('display.max_colwidth', 40)
scaler_df.select("vectorized_features","features").toPandas().head(5)


# ## 2.6 K Means Cluster

# In[40]:


from pyspark.ml.clustering import KMeans
KMeans_algo=KMeans(featuresCol='features', k=4)
KMeans_fit=KMeans_algo.fit(scaler_df)
df11=KMeans_fit.transform(scaler_df)


# In[41]:


df111 = df11.withColumnRenamed('prediction', 'k_cluster')


# In[42]:


df111.toPandas().head(5)


# In[43]:


df111 = df111.drop('features')


# In[44]:


df111 = df111.drop('vectorized_features')


# In[45]:


assembler2 = VectorAssembler()\
         .setInputCols (['age',
                         'duration',
                         'campaign',
                         'previous',
                         'jobencoded',
                         
                         'cons_price_idx',
                         'cons_conf_idx',
                         'nr_employed',
                         'duration',
                         'defaultencoded',
                         'housingencoded',
                         'loanencoded',
                         'contactencoded',
                         'day_of_weekencoded',
                         'poutcomeencoded',
                         'pdays_udfencoded',
                         'euribor3m_udfencoded',
                         'education_udfencoded',
                         'marital_encoded',
                         'month_encoded',
                        'k_cluster']).setOutputCol ("vectorized_features")
        
# In case of missing you can skip the invalid ones
assembler_df=assembler2.setHandleInvalid("skip").transform(df111)
assembler_df.toPandas().head()


# In[46]:


scaler = StandardScaler()\
         .setInputCol ("vectorized_features")\
         .setOutputCol ("features")
        
scaler_model=scaler.fit(assembler_df)
scaler_df=scaler_model.transform(assembler_df)
pd.set_option('display.max_colwidth', 40)
scaler_df.select("vectorized_features","features").toPandas().head(5)


# # 3. Train / Test Split

# In[47]:


final_df = scaler_df.select('features', 'label')


# In[48]:


train, test = final_df.randomSplit([0.8, 0.2], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))


# In[49]:


train.groupby("label").count().show()


# # 4. Confusion Matrix

# In[50]:


class_names=[1.0,0.0]
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# # 5. Model Trainning

# ## Model # 1: Logistic Regression

# In[51]:


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=5)
lrModel = lr.fit(train)
predictions_lr = lrModel.transform(test)
#predictions_train = lrModel.transform(train)
predictions_lr.select('label', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)


# In[52]:


from sklearn.metrics import confusion_matrix,precision_score,recall_score,classification_report

y_true = predictions_lr.select("label")
y_true = y_true.toPandas()

y_pred = predictions_lr.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix-Logistic Regression')
print('precision_score :\n',precision_score(y_true,y_pred,pos_label=1))
print('recall_score :\n',recall_score(y_true,y_pred,pos_label=1))
plt.show()


# In[53]:


accuracy = predictions_lr.filter(predictions_lr.label == predictions_lr.prediction).count() / float(predictions_lr.count())
print("Accuracy-Logistic Regression: ",accuracy)


# In[54]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC-Logistic Regression', evaluator.evaluate(predictions_lr))


# ## Model # 2: Decision Tree

# In[55]:


from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label')
dtModel = dt.fit(train)
predictions_dt = dtModel.transform(test)
#predictions_train = lrModel.transform(train)
predictions_dt.select('label', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)


# In[56]:


y_true = predictions_dt.select("label")
y_true = y_true.toPandas()

y_pred = predictions_dt.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix - Decision Tree')
plt.show()


# In[57]:


accuracy = predictions_dt.filter(predictions_dt.label == predictions_dt.prediction).count() / float(predictions_dt.count())
print("Accuracy-Decision Tree : ",accuracy)


# In[58]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC-Decision Tree', evaluator.evaluate(predictions_dt))


# ## Model # 3: Random Forest

# In[59]:


from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', numTrees=10)
rfModel = rf.fit(train)
predictions_rf = rfModel.transform(test)
#predictions_train = lrModel.transform(train)
predictions_rf.select('label', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)


# In[60]:


y_true = predictions_rf.select("label")
y_true = y_true.toPandas()

y_pred = predictions_rf.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix - Random Forest')
plt.show()


# In[61]:


accuracy = predictions_rf.filter(predictions_rf.label == predictions_rf.prediction).count() / float(predictions_rf.count())
print("Accuracy-Random Forest : ",accuracy)


# In[62]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC-Random Forest', evaluator.evaluate(predictions_rf))


# ## Model # 4: Gradient-boosted tree

# In[63]:


from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(featuresCol = 'features', labelCol = 'label', maxIter=10)
gbtModel = gbt.fit(train)
predictions_gbt = gbtModel.transform(test)
#predictions_train = lrModel.transform(train)
predictions_gbt.select('label', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)


# In[64]:


y_true = predictions_gbt.select("label")
y_true = y_true.toPandas()

y_pred = predictions_gbt.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix-Gradient-boosted tree')
plt.show()


# In[65]:


accuracy = predictions_gbt.filter(predictions_gbt.label == predictions_gbt.prediction).count() / float(predictions_gbt.count())
print("Accuracy-Gradient-boosted tree : ",accuracy)


# In[66]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC-Gradient-boosted tree', evaluator.evaluate(predictions_gbt))


# ## Model # 5: Factorization machines classifier

# In[67]:


from pyspark.ml.classification import FMClassifier
fm = FMClassifier(labelCol="label", featuresCol="features", stepSize=0.001)

# Train model.
fmModel = fm.fit(train)

# Make predictions.
predictions_fm = fmModel.transform(test)

# Select example rows to display.
predictions_fm.select('label', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)


# In[68]:


y_true = predictions_fm.select("label")
y_true = y_true.toPandas()

y_pred = predictions_fm.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix-Factorization machines classifier')
plt.show()


# In[69]:


accuracy = predictions_fm.filter(predictions_fm.label == predictions_fm.prediction).count() / float(predictions_fm.count())
print("Accuracy-Factorization machines classifier : ",accuracy)


# In[70]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC-Factorization machines classifier', evaluator.evaluate(predictions_fm))


# ## 6. Model Comparision
# 
# The best supervised model is Gradient-boosted tree with 93.07% AUC and 90.86% accuracy.
# 
# Gradient-boosted tree model will be chosen as the champion since it performed best compared with other four models

# ## 7. Best Model Saving and Load for future use

# In[71]:


save_path = "D:\model"


# In[72]:


os.makedirs(save_path, exist_ok=True)


# In[73]:


from pyspark.ml.classification import GBTClassifier
# gbt = GBTClassifier(featuresCol = 'features', labelCol = 'label', maxIter=10)
# gbtModel = gbt.fit(train)
gbtModel.write().overwrite().save(save_path)


# In[ ]:


gbtModel = gbtModel.load(save_path)


# ## 8. Prescriptive Recommendations
# Based on this analysis, we have seen a higher subscribe rate among
# 
# * Highly educated
# * Single people
# * Special months (Mar. & Sep. & Otc. & Dec)
# 
# To capitalize on that, we should focus advertising on that demographic with targeted advertising in special months (Mar. & Sep. & Otc. & Dec), highly educated and single people!
