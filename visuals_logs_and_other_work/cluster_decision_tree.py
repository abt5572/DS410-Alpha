#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyspark
import pandas as pd
import numpy as np
import math

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import PCA
import matplotlib.pyplot as plt

from pyspark.sql.functions import col, mean, column
import matplotlib.pyplot as plt
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row
#from pyspark.mllib.recommendation import ALS

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.util import MLUtils

from decision_tree_plot.decision_tree_parser import decision_tree_parse
from decision_tree_plot.decision_tree_plot import plot_trees


# In[1]:


#from IPython.core.display import HTML
#display(HTML("<style>pre { white-space: pre !important; }</style>"))


# In[3]:


ss = SparkSession.builder.config("spark.driver.memory", "16g").appName("ProjectTree1").getOrCreate()
#ss = SparkSession.builder.config("spark.driver.memory", "5g").master("local").appName("PCAExample1").getOrCreate()
#ss = SparkSession.builder.master("local").appName("PCAExample1").getOrCreate()


# In[4]:


ss.sparkContext.setCheckpointDir("/storage/home/sxs6549/work/Project/scratch")


# In[5]:


#%%time
df_raw = ss.read.csv("wildfiredb.csv", header=True, inferSchema=True)
#df_raw = spark.read.csv("wildfire100.csv" , header = True, inferSchema = True)
#column_names = df_raw.columns

#df_raw = df_raw.drop("acq_date")
df_raw = df_raw.dropna()


# In[4]:


#%%time
#df_raw_trial = ss.read.csv("fire_small.csv", header=True, inferSchema=True)
#df_raw = spark.read.csv("wildfire100.csv" , header = True, inferSchema = True)
#column_names = df_raw_trial.columns

#df_raw = df_raw.drop("acq_date")
#df_raw_trial = df_raw_trial.dropna()


# In[4]:


#col_list = list(df_raw.columns)
#col_list


# In[5]:


#col_list_new = list(set(col_list) - set(['_c0', 'Polygon_ID', 'acq_date', 'frp']))
#col_list_new


# In[6]:


#feature_columns = df_raw.columns
#col_list = list(df_raw.columns)
col_list = list(df_raw.columns)
feature_inputs = list(set(col_list) - set(['_c0', 'Polygon_ID', 'acq_date', 'frp']))

assembler_tree = VectorAssembler(inputCols = feature_inputs, outputCol =  "features")
assembled_data_tree = assembler_tree.transform(df_raw)
#scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
#scaler_model = scaler.fit(assembled_data)

#scaled_data = scaler_model.transform(assembled_data)


# In[7]:


pca_tree = PCA(k=36, inputCol="features", outputCol="pcaFeatures")
model_tree = pca_tree.fit(assembled_data_tree)
result_tree = model_tree.transform(assembled_data_tree)


# In[19]:


from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.util import MLUtils

from decision_tree_plot.decision_tree_parser import decision_tree_parse
from decision_tree_plot.decision_tree_plot import plot_trees
# Split the data into training and test sets (20% held out for testing)
(trainingData, testingData) = result_tree.randomSplit([0.8, 0.2], seed=1237)

#Code cell for Part 7
## Initialize a Pandas DataFrame to store evaluation results of all combination of hyper-parameter settings
hyperparams_eval_df = pd.DataFrame( columns = ['max_depth', 'minInstancesPerNode', 'training_rmse', 'testing_rmse',  'Best Model'] )
# initialize index to the hyperparam_eval_df to 0
index =0 
# initialize lowest_error
lowest_testing_rmse = 100000
# Set up the possible hyperparameter values to be evaluated
max_depth_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
minInstancesPerNode_list = [2, 3, 4, 5, 6, 7]
#max_depth_list = [2]
#minInstancesPerNode_list = [9]
#labelIndexer = StringIndexer(inputCol="class", outputCol="indexedLabel").fit(data2)
#feature_inputs = list(set(col_list) - set(['_c0', 'Polygon_ID', 'acq_date', 'frp']))
#assembler = VectorAssembler( inputCols=feature_inputs, outputCol="features")
#labelConverter = IndexToString(inputCol = "prediction", outputCol="predictedClass", labels=labelIndexer.labels)
model_path="/storage/home/sxs6549/work/Project/fire1_DTmodel_vis"


# In[20]:


#%%time
for max_depth in max_depth_list:
    for minInsPN in minInstancesPerNode_list:
        trainingData.persist()
        testingData.persist()
        
        seed = 37
        # Construct a DT model using a set of hyper-parameter values and training data
        #dt= DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features", maxDepth= max_depth, minInstancesPerNode= minInsPN)
        dt = DecisionTreeRegressor(labelCol="frp", featuresCol="pcaFeatures", maxDepth=max_depth, minInstancesPerNode=minInsPN)
        #pipeline = Pipeline(stages=[labelIndexer, assembler, dt, predictionConverter])
        model = dt.fit(trainingData)
        training_predictions = model.transform(trainingData)
        testing_predictions = model.transform(testingData)
        #evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
        evaluator = RegressionEvaluator(labelCol="frp", predictionCol="prediction", metricName="rmse")
        training_rmse = evaluator.evaluate(training_predictions)
        testing_rmse = evaluator.evaluate(testing_predictions)
        # We use 0 as default value of the 'Best Model' column in the Pandas DataFrame.
        # The best model will have a value 1000
        hyperparams_eval_df.loc[index] = [ max_depth, minInsPN, training_rmse, testing_rmse, 0]  
        index = index +1
        if testing_rmse < lowest_testing_rmse :
            best_max_depth = max_depth
            best_minInsPN = minInsPN
            best_index = index -1
            best_parameters_training_rmse = training_rmse
            best_DTmodel= model
            best_tree = decision_tree_parse(best_DTmodel, ss, model_path)
            column = dict( [ (str(idx), i) for idx, i in enumerate(feature_inputs) ])           
            lowest_testing_rmse = testing_rmse
print('The best max_depth is ', best_max_depth, ', best minInstancesPerNode = ',       best_minInsPN, ', testing rmse = ', lowest_testing_rmse) 
column = dict([(str(idx), i) for idx, i in enumerate(feature_inputs)])


# In[28]:


#training_predictions.show(100)


# In[15]:


#Code cell for Part 7
best_model_path_part7="/storage/home/sxs6549/work/Project/fire1_DT_HPT_cluster"


# In[16]:


#Code cell for Part 7
best_tree=decision_tree_parse(best_DTmodel, ss, best_model_path_part7)
column = dict([(str(idx), i) for idx, i in enumerate(feature_inputs)])
plot_trees(best_tree, column = column, output_path = '/storage/home/sxs6549/work/Project/fire1_DT_HPT_cluster.html')


# In[17]:


#Code cell for Part 7
# Store the Testing RMS in the DataFrame
hyperparams_eval_df.loc[best_index]=[best_max_depth, best_minInsPN, best_parameters_training_rmse, lowest_testing_rmse, 1000]


# In[18]:


output_path = "/storage/home/sxs6549/work/Project/fire1_HPT_cluster.csv"
hyperparams_eval_df.to_csv(output_path)  


# In[ ]:


ss.stop()


# In[ ]:




