import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import scipy 
import math  
import sklearn
import random
import csv
<<<<<<< HEAD
import time


#Important Variables: 
Learning_Rate = 0.0083
=======


#Important Variables: 
Learning_Rate = 0.000001
Bias = 0.008
>>>>>>> fb597d00077e61684a66b4b091751afa03d8499f


random.seed(1)


#Functions:
def Weight_Vector_Gen(dimension_of_vector):
	Weight_Vector_IN = np.zeros((dimension_of_vector,1))
	for i in range(len(Weight_Vector_IN)):
<<<<<<< HEAD
		t = float((random.randint(-50,50))/100)
=======
		t = float((random.randint(-10,10))/100)
>>>>>>> fb597d00077e61684a66b4b091751afa03d8499f
		Weight_Vector_IN[i] = t
	return Weight_Vector_IN

DataFrame_Dict = {}
def Sub_DataFrame_Train_According_to_label(DataFrame, Label_Value):
 	DataFrame = pd.DataFrame(DataFrame)
 	newDataFrame = DataFrame[DataFrame.label==Label_Value]
 	return newDataFrame

<<<<<<< HEAD
def Softmax(z):
	return np.exp(-z)/np.sum(np.exp(-z))

=======
>>>>>>> fb597d00077e61684a66b4b091751afa03d8499f

#Initialization of Weight vectors for each layer.

random.seed(1)
Hidden_Layer = [None]*12
Output_Layer = [None]*10
<<<<<<< HEAD
Delta_Hidden_to_Output_Vector = [None]*10
Delta_Input_to_Hidden_Vector = [None]*12


#Initialization of necessary vectors 

Weighted_Sums_Hidden_Layer = np.zeros((12,1))
Weighted_Sums_Output_Layer = np.zeros((10,1))
Weighted_Input_to_Hidden_Vector = np.zeros((10,1))

Sigma_Output_Layer_Vector = np.zeros((10,1))
Sigma_Input_to_Hidden_Vector = np.zeros((12,1))

Delta_Hidden_to_Output_Sub_Vector = np.zeros((12,1))
Delta_Input_to_Hidden_Sub_Vector = np.zeros((784,1))


#Filling of Weights Vectors
for i in range(len(Hidden_Layer)):
	Hidden_Layer[i] = Weight_Vector_Gen(784)

=======
for i in range(len(Hidden_Layer)):
	Hidden_Layer[i] = Weight_Vector_Gen(784)


>>>>>>> fb597d00077e61684a66b4b091751afa03d8499f
for i in range(len(Output_Layer)):
	Output_Layer[i] = Weight_Vector_Gen(12)


<<<<<<< HEAD
#Filling of Delta Vector for Hidden Layer
for i in range(len(Delta_Input_to_Hidden_Vector)):
	Delta_Input_to_Hidden_Vector[i] = np.zeros((784,1))
for i in range(len(Delta_Hidden_to_Output_Vector)):
	Delta_Hidden_to_Output_Vector[i] = np.zeros((12,1))

#Initialization of training set
TRAIN = pd.read_csv('mnist_train.csv', delimiter = ',')

#Initilization of one-hot encoding
pd.to_numeric(TRAIN['label'])
possible_outputs = list(TRAIN['label'].unique())
possible_outputs = sorted(possible_outputs)
one_hot_encoded = pd.get_dummies(possible_outputs).values

for x in range(0, 200):
	for a in range(0, 30):
		
		#Target result (one-hot-encoding used)
		Target = one_hot_encoded[TRAIN['label'][a]].reshape((10,1))

		#For each training instance, propagate forward all activation results from input-layer to hidden-layer.
		#Hidden-Layer nodes possess 784 weight-vector. 
		for i in range(0, len(Hidden_Layer)):
			Input_Layer_Output = np.dot(Hidden_Layer[i].transpose(), (TRAIN.loc[a:a, '1x1':'28x28'].transpose().values)/255)
			Input_Layer_Activation = 1 / (1 + np.exp(-Input_Layer_Output)) + float((random.randint(-20,20))/100)
			Weighted_Sums_Hidden_Layer[i] = (Input_Layer_Activation)
		
		#For each hidden-node, propagate forward activation results to output-layer. 
		#Output-layer nodes possess 12 weight vector. 
		for j in range(0, len(Output_Layer)):	
			Hidden_Layer_Output = np.dot(Output_Layer[j].transpose(), Weighted_Sums_Hidden_Layer) 
			Weighted_Sums_Output_Layer[j] = Hidden_Layer_Output + float((random.randint(-20,20))/100)
		
		Weighted_Sums_Output_Layer = Softmax(Weighted_Sums_Output_Layer)
			# Hidden_Layer_Activation = 1 / (1 + np.exp(-Hidden_Layer_Output)) + float((random.randint(-10,10))/100)
			# Weighted_Sums_Output_Layer[j] = (Hidden_Layer_Activation)
		

		#Start of backpropagation. 

		#Compute Error on Output-Layer
		Error_Out_Vector = Target - Weighted_Sums_Output_Layer

		New_Global_Error=sum(Error_Out_Vector*Error_Out_Vector)

	#Compute Sigma and Weight-Delta from Output Layer to Hidden Layer.
		for k in range(0, len(Output_Layer)):
			#Fill in Sigma Vector
			Sigma_Output_Layer_Vector[k] = Error_Out_Vector[k]*(1-Weighted_Sums_Output_Layer[k])*(Weighted_Sums_Output_Layer[k])
			for h in range(0, len(Hidden_Layer)):
			#Create Delta Vector for Weight update between Hidden and Output Layer
				Delta_Hidden_to_Output_Sub_Vector[k] = Learning_Rate*Sigma_Output_Layer_Vector[k]*Weighted_Sums_Hidden_Layer[h]
			
			Delta_Hidden_to_Output_Vector[k] = Delta_Hidden_to_Output_Sub_Vector
			#Multiply Output Layer Delta Node with each Weight connected to that node, and then sum it up. 
			Weighted_Sum_Error = sum(Sigma_Output_Layer_Vector[k]*Output_Layer[k])
			Weighted_Input_to_Hidden_Vector[k] = Weighted_Sum_Error 

		#Sigma and Delta Generation for Hidden Layer nodes 	
		for h in range(0, len(Hidden_Layer)):
			for j in range(0, len(Output_Layer)):
				Sigma_Input_to_Hidden_Vector[h] = Weighted_Sums_Hidden_Layer[h]*(1-Weighted_Sums_Hidden_Layer[h])*(Weighted_Input_to_Hidden_Vector[j])
			
			Delta_Input_to_Hidden_Sub_Vector = Learning_Rate*Sigma_Input_to_Hidden_Vector[h]*((TRAIN.loc[a:a, '1x1':'28x28'].values)/255).transpose())
			Delta_Input_to_Hidden_Vector[h] = Delta_Input_to_Hidden_Sub_Vector

		#Weight updating: Final. 

		for o in range(0,len(Output_Layer)):
			Output_Layer[o] = Output_Layer[o] + Delta_Hidden_to_Output_Vector[o]

		for h in range(0, len(Hidden_Layer)):
			Hidden_Layer[h] = Hidden_Layer[h] + Delta_Input_to_Hidden_Vector[h]
		


#END OF ERROR PROPAGATION





#Testing 
time.sleep(5)

print('Start of Testing')

GoodCount = 0

for a in range(10001, 11000):

	Test_Target = one_hot_encoded[TRAIN['label'][a]].reshape((10,1))
	for i in range(len(Hidden_Layer)):
		Input_Layer_Output = np.dot(Hidden_Layer[i].transpose(), TRAIN.loc[a:a, '1x1':'28x28'].transpose().values)
		Input_Layer_Activation = 1 / (1 + np.exp(-Input_Layer_Output)) + float((random.randint(-20,20))/100)
		Weighted_Sums_Hidden_Layer[i] = (Input_Layer_Activation)
	
	#For each hidden-node, propagate forward activation results to output-layer. 
	#Output-layer nodes possess 12 weight vector. 
	for j in range(len(Output_Layer)):	
		Hidden_Layer_Output = np.dot(Output_Layer[j].transpose(), Weighted_Sums_Hidden_Layer) 
		# Hidden_Layer_Activation = 1 / (1 + np.exp(-Hidden_Layer_Output)) + float((random.randint(-10,10))/100)
		Weighted_Sums_Output_Layer[j] = (Hidden_Layer_Output) + float((random.randint(-20,20))/100)
	Weighted_Sums_Output_Layer = Softmax(Weighted_Sums_Output_Layer) 

	# print(np.argmax(Weighted_Sums_Output_Layer),np.argmax(Test_Target))

	if(np.argmax(Weighted_Sums_Output_Layer)==(np.argmax(Test_Target))):
		GoodCount+=1

	
Success_Rate = GoodCount/1000
=======
#Initialization of testcases

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

>>>>>>> fb597d00077e61684a66b4b091751afa03d8499f
print(Success_Rate)
