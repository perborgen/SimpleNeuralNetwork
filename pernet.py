import numpy as np
import random, math
import pandas as pd

df = pd.read_csv('data.csv',header=0)

X = np.array(df[["grade1","grade2"]])
y = np.array(df[["label"]])


Y_train = y[0:70]
Y_test  = y[70:]

X_train = X[0:70]
X_test  = X[70:]


#np.random.seed(1)

def sigmoid(s):
	return 1/(1+np.exp(-s))

def sigmoidDeriv(s):
	return s*(1 - s)

weights_0 = 2*np.random.rand(2,3) - 1
weights_1 = 2*np.random.rand(3,1) - 1

def forward(weights_0,weights_1,X):
	l0 = X
	input_l1 = np.dot(l0,weights_0)
	l1 = sigmoid(input_l1)
	input_l2 = np.dot(l1,weights_1)
	l2 = sigmoid(input_l2)	
	return l0, l1, l2

def train(weights_0,weights_1):
	for j in xrange(80001):
		l0, l1, l2 = forward(weights_0,weights_1,X_train)
		l2_error = Y_train - l2
		l2_delta = l2_error*sigmoidDeriv(l2) 
		l1_output_delta = l2_delta.dot(weights_1.T)
		l1_input_delta = l1_output_delta * sigmoidDeriv(l1)
		weights_0_delta = l0.T.dot(l1_input_delta)
		weights_1_delta = l1.T.dot(l2_delta)
		weights_0 += weights_0_delta
		weights_1 += weights_1_delta
		if (j% 10000) == 0:
			print "Error:" + str(np.mean(np.abs(l2_error)))
	return weights_0,weights_1



w_0, w_1 = train(weights_0,weights_1)
n,nn,raw_predictions = forward(w_0,w_1,X_test)
raw_answers = Y_test

predictions =[]
answers = []
# cleanup predictions (not whole integers)
for pred in raw_predictions:
	if pred <= 0.5:
		predictions.append(0)
	elif pred > 0.5:
		predictions.append(1)

# cleanup answers (2D array to normal array)
for pred in raw_answers:
	if pred <= 0.5:
		answers.append(0)
	elif pred > 0.5:
		answers.append(1)

def check_results(predictions,answers):
	hits = 0
	for k in xrange(len(predictions)):
		p = predictions[k]
		a = answers[k]
		if a == p:
			hits+=1
	
	hit_ratio = float(hits)/len(predictions)
	return hit_ratio

ratio = check_results(predictions, answers)

print 'ratio: ', ratio





