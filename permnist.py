import numpy as np
import random, math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

df = pd.read_csv('mnist.csv',header=0)
y = df["label"]
X = df.drop("label", axis=1)
y = np.array(y)
X = np.array(X)
y_train = y[0:2000]
X_train = X[0:2000] / float(256)
y_test = y[2001:2500]
X_test = X[2001:2500]/ float(256)



def create_label_array(num):
	label_array = []
	i = 0
	while i < 10:
		if num == i:
			label_array.append(1)
		else:
			label_array.append(0)
		i += 1
	return label_array

def create_int(arr):
	i=0
	while i < len(arr):
		if arr[i] == 1:
			return i
		i+=1
	return 'X'

def sigmoid(s):
	return 1/(1+np.exp(-s))

def sigmoidDeriv(s):
	return s*(1 - s)

w_0 = 2*np.random.rand(784,30) - 1
w_1 = 2*np.random.rand(30,10) - 1
b_0 = 2*np.random.rand(30) - 1
b_1 = 2*np.random.rand(10) - 1

def forward(weights_0,weights_1,bias_0,bias_1,X):
	l0 = X
	#print 'bias_1', bias_1
	input_l1 = np.dot(l0,weights_0) + bias_0
	l1 = sigmoid(input_l1)
	#print 'before: ', np.dot(l1,weights_1)
	input_l2 = np.dot(l1,weights_1) + bias_1
	#print 'input_l2', input_l2
	#print 'input_l2.shape', input_l2.shape

	l2 = sigmoid(input_l2)
	return l0, l1, l2

def sum_sq_errors(errors):
	length = len(errors)
	sum = np.sum(errors)
	sQsum = sum **2
	halfsQsum = sQsum/(2*length)
	return halfsQsum

def sq_errors_derivative(predictions,labels):
	length = len(predictions)
	sumPredictions = np.sum(predictions)
	sumLabels = np.sum(labels)
	result = sumPredictions - sumLabels
	return result/length

def cost(y, t):
	sumSq = ((t - y)**2).sum()
	result = sumSq * 0.5 / len(y)
	return result

def cost_deriv(predictions,y):
	return (predictions-y)


def train(weights_0,weights_1,bias_0,bias_1,train_labels):
	alpha = 0.2
	bias_alpha = 0.1
	for j in xrange(100000):
		l0, l1, l2 = forward(weights_0,weights_1,bias_0,bias_1,X_train)
		l2_error = cost_deriv(l2,train_labels)
		l2_delta = l2_error * sigmoidDeriv(l2) 
		l1_output_delta = l2_delta.dot(weights_1.T)
		l1_input_delta = l1_output_delta * sigmoidDeriv(l1)
#		print 'l2_error.shape: ', l2_error.shape
#		print 'l2_delta.shape: ', l2_delta.shape
#		print 'l1_output_delta.shape: ', l1_output_delta.shape
		#print 'l1_input_delta.shape', l1_input_delta.shape
		#print 'l0.T.shape', l0.T.shape
		weights_0_delta = l0.T.dot(l1_input_delta) / len(X_train)
		weights_1_delta = l1.T.dot(l2_delta) / len(X_train)
		weights_0 -= (weights_0_delta * alpha)
		weights_1 -= (weights_1_delta * alpha)
#		print 'weights_0_delta.shape: ', weights_0_delta.shape
#		print 'weights_0.shape: ', weights_0.shape
#		print 'l2_delta.shape: ', l2_delta.shape
#		print 'bias_1.shape: ', bias_1.shape
#		print 'l2_delta.sum(axis=0)',l2_delta.sum(axis=0)
		bias_0 -= (l1_input_delta.sum(axis=0) * bias_alpha)
		bias_1 -= (l2_delta.sum(axis=0) * bias_alpha)
		if (j% 1000) == 0:
			c = cost(train_labels,l2)
			print 'error: ',c
			print 'bias_1: ', bias_1
			#print 'weights_1_delta: ', weights_1_delta
	return weights_0,weights_1,bias_0,bias_1

new_y_train = []
for label_num in y_train:
	label_array = create_label_array(label_num)
	new_y_train.append(label_array)
y_train_arrays = np.array(new_y_train)

w0, w1, b0, b1 = train(w_0,w_1,b_0,b_1,y_train_arrays)

l0, l1, pred = forward(w0,w1,b_0,b_1,X_test)

print 'pred: ', pred
pred = pred.round(decimals=0)

predictions = []


for x in pred:
	answer = create_int(x)
	predictions.append(answer)



print 'perdictions: '
print predictions
print 'labels on test: '
print y_test


def check_results(predictions,answers):
	hits = 0
	for k in xrange(len(predictions)):
		p = predictions[k]
		a = answers[k]
		if a == p:
			hits+=1
	
	hit_ratio = float(hits)/len(predictions)
	return hit_ratio

ratio = check_results(predictions, y_test)

print 'ratio: ', ratio





