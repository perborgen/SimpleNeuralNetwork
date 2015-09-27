import numpy as np


X = np.array([[0,0,1],
			  [0,1,1],
			  [1,0,1],
			  [1,1,1]])

y = np.array([[0],
			  [1],
			  [1],
			  [0]])

np.random.seed(1)

def sigmoid(s):
	return 1/(1+np.exp(-s))

def sigmoidDeriv(s):
	return s*(1 - s)

weights_0 = 2*np.random.rand(3,4) - 1
weights_1 = 2*np.random.rand(4,1) - 1

for j in xrange(10001):
	#forward 
	l0 = X
	#print 'weights_0'
	#print weights_0
	#print 'l0'
	#print l0
	input_l1 = np.dot(l0,weights_0)
	#print 'input_l1'
	#print input_l1
	l1 = sigmoid(input_l1)
	#print 'l1'
	#print l1
	input_l2 = np.dot(l1,weights_1)
	#print input_l2
	l2 = sigmoid(input_l2)
	#print l2


	#backward
	l2_error = y - l2
	l2_delta = l2_error*sigmoidDeriv(l2) 

	l1_output_delta = l2_delta.dot(weights_1.T)
	l1_input_delta = l1_output_delta * sigmoidDeriv(l1)

	weights_0_delta = l0.T.dot(l1_input_delta)
	weights_1_delta = l1.T.dot(l2_delta)

	weights_0 += weights_0_delta
	weights_1 += weights_1_delta

	if (j% 10000) == 0:
		print "Error:" + str(np.mean(np.abs(l2_error)))

	#print 'l2_delta: '
	#print l2_delta
	#print 'weights_1: '
	#print weights_1
	#print 'l1_output_delta:'
	#print l1_output_delta
	#print 'sigmoidDeriv(l1)'
	#print sigmoidDeriv(l1)
	#print 'l1_input_delta: '
	#print l1_input_delta
	#print 'l0'
	#print l0


















