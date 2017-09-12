import numpy as np 
import matplotlib.pyplot as plt 
from testCases_v2 import *
from dnn_utils_v2 import sigmoid,sigmoid_backward,relu,relu_backward

plt.rcParams["figure.figsize"] = (5.0,4.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

np.random.seed(1)

#(1.1)Initalize_parameters:One hidden layer
def initalize_parameters(n_x,n_h,n_y):
	W1 = np.random.randn(n_h,n_x) * 0.01
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(n_y,n_h) * 0.01
	b2 = np.zeros((n_y,1))

	assert( W1.shape == (n_h, n_x) )
	assert( b1.shape == (n_h, 1) )
	assert( W2.shape == (n_y, n_h) )
	assert( b2.shape == (n_y, 1) )
	
	parameters = {
		"W1": W1,
		"b1": b1,
		"W2": W2,
		"b2": b2
	}
    
	return parameters  

'''
parameters = initalize_parameters(3,2,1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''

#(1.2)Initalize_parameters:L-layer
def initialize_parameters_deep(layer_dims):
	np.random.randn(3)
	parameters = {}
	L = len(layer_dims)

	for l in range(1,L):
		parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
		parameters["b" + str(l)] = np.zeros((layer_dims[l],1))

		assert( parameters["W" + str(l)].shape == (layer_dims[l],layer_dims[l-1]) )
		assert( parameters["b" + str(l)].shape == (layer_dims[l],1) )

	return parameters

'''
parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''

#(2.1)linear_forward(A, W, b);Return Z,cache;For one layer
def linear_forward(A, W, b):
	Z = np.dot(W,A) + b

	assert( Z.shape == (W.shape[0],A.shape[1]) )
	cache = (A,W,b)
	return Z,cache

'''
A, W, b = linear_forward_test_case()
Z, linear_cache = linear_forward(A, W, b)
print("Z = " + str(Z))
'''


#(2.2)linear_activation_forward;Return A,cache
def linear_activation_forward(A_prev, W, b, activation):
	if activation == "sigmoid":
		Z,linear_cache = linear_forward(A_prev,W,b)
		A,activation_cache = sigmoid(Z)

	if activation == "relu":
		Z,linear_cache = linear_forward(A_prev,W,b)
		A,activation_cache = relu(Z)

	assert(A.shape == (W.shape[0],A_prev.shape[1]) )
	cache = (linear_forward,activation_cache)

	return A,cache

'''
A_prev, W, b = linear_activation_forward_test_case()
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))
'''

#(2.3)L_model_forward：Return AL,caches
def L_model_forward(X,parameters):
	caches = []
	A = X
	L = len(parameters) // 2

	for l in range(1,L):
		A_prev = A
		A,cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
		caches.append(cache)

	AL,cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
	caches.append(cache)

	assert( AL.shape == (1,X.shape[1]) )
	return AL,caches

'''
X, parameters = L_model_forward_test_case()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))
'''


#(3.1)Cost Function
def compute_cost(AL,Y):
	m = Y.shape[1]

	cost = 1/m * -( np.dot(np.log(AL),Y.T) + np.dot(np.log(1-AL),(1-Y).T) )
	cost = np.squeeze(cost)
	return cost

'''
Y, AL = compute_cost_test_case()
print("cost = " + str(compute_cost(AL, Y)))
'''

#(4.1)Linear backward:Return dA_prev,dW,db
def linear_backward(dZ,cache):
	A_prev,W,b = cache 
	m = A_prev.shape[1]

	dW = np.dot(dZ,A_prev.T) / m
	db = np.sum(dZ,axis=1,keepdims=True) /m
	dA_prev = np.dot(W.T,dZ)

	assert(dW.shape == W.shape)
	assert(db.shape == b.shape)
	assert(dA_prev.shape == A_prev.shape)

	return dA_prev,dW,db


'''
dZ, linear_cache = linear_backward_test_case()
dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
'''

#(4.2)Linear activation backward:Return dA_prev,dW,db
def linear_activation_backward(dA,cache,activation):
	linear_cache,activation_cache = cache

	if activation == "relu":
		dZ = relu_backward(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,linear_cache)

	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,linear_cache)

	return dA_prev,dW,db

'''
AL, linear_activation_cache = linear_activation_backward_test_case()
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
'''

#(4.3)L_Model backward
def L_model_backward(AL,Y,caches):

	grads = {}
	L = len(caches)
	m = AL.shape[1]
	Y.reshape(AL.shape)

	dAL = -(Y/AL) + (1-Y)/(1-AL)

	current_cache = caches[L-1]
	grads["dA" + str(L)],grads["dW" + str(L)],grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")

	for l in range(L-1)[::-1]:
		current_cache = caches[l]
		dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache,"relu")
		grads["dA" + str(l+1)] = dA_prev_temp
		grads["dW" + str(l+1)] = dW_temp
		grads["db" + str(l+1)] = db_temp

		return grads

'''
AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dA1 = "+ str(grads["dA1"]))
'''

#(5.1)update parameters
def update_parameters(parameters,grads,learning_rate):
	L = len(parameters) // 2

	for l in range(L):
		parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*grads["dW" + str(l+1)]
		parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*grads["db" + str(l+1)]

	return parameters

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))