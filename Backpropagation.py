import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import scipy 
import math  
import sklearn
import random
import csv


#Important Variables: 
Learning_Rate = 0.000001
Bias = 0.008


random.seed(1)


#Functions:
def Weight_Vector_Gen(dimension_of_vector):
	Weight_Vector_IN = np.zeros((dimension_of_vector,1))
	for i in range(len(Weight_Vector_IN)):
		t = float((random.randint(-10,10))/100)
		Weight_Vector_IN[i] = t
	return Weight_Vector_IN

DataFrame_Dict = {}
def Sub_DataFrame_Train_According_to_label(DataFrame, Label_Value):
 	DataFrame = pd.DataFrame(DataFrame)
 	newDataFrame = DataFrame[DataFrame.label==Label_Value]
 	return newDataFrame


#Initialization of Weight vectors for each layer.

random.seed(1)
Hidden_Layer = [None]*12
Output_Layer = [None]*10
for i in range(len(Hidden_Layer)):
	Hidden_Layer[i] = Weight_Vector_Gen(784)


for i in range(len(Output_Layer)):
	Output_Layer[i] = Weight_Vector_Gen(12)


#Initialization of training set

TRAIN = pd.read_csv('mnist_train.csv', delimiter = ',')
pd.to_numeric(TRAIN['label'])

Weighted_Sums_Hidden_Layer = np.zeros((12,1))
Weighted_Sums_Output_Layer = np.zeros((10,1))

GoodCount = 0
for a in range(1, 20000):
	for i in range(len(Hidden_Layer)):

		Input_Layer_Output = np.dot(Hidden_Layer[i].transpose(), TRAIN.loc[a:a, '1x1':'28x28'].transpose().values)
		
		Input_Layer_Activation = 1 / (1 + np.exp(-Input_Layer_Output))
		
		Weighted_Sums_Hidden_Layer[i] = (Input_Layer_Activation)
	
	for j in range(len(Output_Layer)):
		
		Hidden_Layer_Output = np.dot(Output_Layer[j].transpose(), Weighted_Sums_Hidden_Layer) 
			
		Hidden_Layer_Activation = 1 / (1 + np.exp(-Hidden_Layer_Output))
		
		Weighted_Sums_Output_Layer[j] = (Hidden_Layer_Activation)
	
	# print(Weighted_Sums_Output_Layer)

	# print(np.argmax(Weighted_Sums_Output_Layer))
	# print(TRAIN['label'][a])

	##Error Processing

	for k in range(len(Output_Layer)):
		

		Target_Error = TRAIN['label'][a] - Weighted_Sums_Output_Layer[k]
		
		Sigma_Output_Layer = Target_Error*(1-Weighted_Sums_Output_Layer[k])*(Weighted_Sums_Output_Layer[k])
		
		Weighted_Sum_errors = sum(Sigma_Output_Layer*Output_Layer[k]) + Bias

		
		# Delta_Output_Layer = Learning_Rate*Sigma_Output_Layer*Weighted_Sums_Hidden_Layer

		# Weighted_Sum_errors = np.dot(Output_Layer[j].transpose(), Delta_Output_Layer)

		# Output_Layer[k] = Output_Layer[k] + Delta_Output_Layer
		
	for z in range(len(Hidden_Layer)):
		
		Delta_Output_Layer = Learning_Rate*Sigma_Output_Layer*Weighted_Sums_Hidden_Layer[z]

		Sigma_Hidden_Layer = Target_Error*Weighted_Sums_Output_Layer[k]*Weighted_Sum_errors

		Delta_Hidden_Layer = Learning_Rate*Sigma_Hidden_Layer*Weighted_Sums_Hidden_Layer[z]

		Hidden_Layer[z] = Hidden_Layer[z] + Delta_Hidden_Layer

	# for k in range(len(Output_Layer)):
		
	# 	Output_Layer[k] = Output_Layer[k] + Delta_Output_Layer

	# print(np.argmax(Weighted_Sums_Output_Layer))
	if(np.argmax(Weighted_Sums_Output_Layer)==TRAIN['label'][a]):
		GoodCount+=1

Success_Rate = GoodCount/20000

print(Success_Rate)