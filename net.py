# 'http://iamtrask.github.io/2015/07/12/basic-python-network/'
import numpy as np

def nonlin(x,deriv=False):
	if(deriv == True):
		return x*(1-x)
	return 1/(1+np.exp(-x))


X=np.array([[0,0,1],
			[0,1,1],
			[1,0,1],
			[1,1,1]])

y = np.array([[0,0,1,1]]).T

#np.random.seed(1)

# weights
syn0 = 2*np.random.random((3,1)) - 1
print 'syn0 start:'
print syn0




for iter in xrange(1):
	# forward propagation
	l0 = X
	l0_before = np.dot(l0,syn0)
	print 'l0_before: '
	print l0_before
	l1 = nonlin(l0_before,deriv=False)
	print 'l1: '
	print l1
	# how much did we miss?
	l1_error = y - l1
	l1_delta = l1_error*nonlin(l1,True)

	print 'l0.T:'
	print l0.T
	print 'l1_delta:'
	print l1_delta
	weight_delta = np.dot(l0.T, l1_delta)
	print 'weight_delta: '
	print weight_delta
	syn0 += weight_delta
	print 'updated syn0:'
	print syn0

print "Output After Training:"
print l1