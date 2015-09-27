import numpy as np

def nonlin(x,deriv=False):
	if(deriv == True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

X = np.array([[0,0,1],
			  [0,1,1],
			  [1,0,1],
			  [1,1,1]])

y = np.array([[0],
			  [1],
			  [1],
			  [0]])

np.random.seed(1)

syn0 = 2 * np.random.random((3,4))-1
syn1 = 2 * np.random.random((4,1))-1 


for j in xrange(10001):
	l0 = X
	input_l1 = np.dot(l0,syn0)
	l1 = nonlin(input_l1)
	input_l2 = np.dot(l1,syn1)
	l2 = nonlin(input_l2)

	#print 'l0:'
	#print l0	
	#print 'syn0: '
	#print syn0
	#print 'inplut_l1: '
	#print input_l1
	#print 'l1'
	#print l1
	#print 'syn1'
	#print syn1
	#print 'l2'
	#print l2


	l2_error = y - l2
	l2_delta = l2_error * nonlin(l2,deriv=True)
	
	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error * nonlin(l1,deriv=True)
	syn0 += l0.T.dot(l1_delta)
	syn1 += l1.T.dot(l2_delta)


	if (j% 10000) == 0:
		print "Error:" + str(np.mean(np.abs(l2_error)))
		print 'l1'
		print l1
		print 'l2_delta:'
		print l2_delta



