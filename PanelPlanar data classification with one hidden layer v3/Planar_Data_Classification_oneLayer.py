import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets


np.random.seed(1)
X,Y = load_planar_dataset()

#Visualize the data
plt.scatter(X[0,:],X[1,:],c=Y,s=40,cmap=plt.cm.Spectral)


#DataSizes
shape_X = X.shape
shape_Y = Y.shape
m = int(Y.shape[1])
print("The shape of X is:" + str(shape_X))
print("The shape of Y is:" + str(shape_Y))
print("I have m = %d training examples" %(m))

#Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T,Y.T)

#Plot the decidion boundary for logistic regression
plot_decision_boundary(lambda x:clf.predict(x),X,Y)
#plt.show()

#Print accuracy
LR_predictions = clf.predict(X.T)
print("Accuracy of logistic regression:%d" %float( (np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) 
	+ "%" + "(percentage of correctly labelled datapoints)")

#Neural Network model
#GRADED FUNCTION:layer_sizes

def layer_sizes(X,Y):
	n_x = X.shape[0]
	n_h = 4
	n_y = Y.shape[0]
	return (n_x,n_h,n_y)

X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))

#GRADED FUNCTION:initialize_parameters
def initialize_parameters(n_x,n_h,n_y):

	W1 = np.random.randn(n_h,n_x) * 0.01
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(n_y,n_h) * 0.01
	b2 = np.zeros((n_y,1))

	assert( W1.shape == (n_h,n_x))
	assert( b1.shape == (n_h,1))
	assert( W2.shape == (n_y,n_h))
	assert( b2.shape == (n_y,1))

	parameters = {
		"W1":W1,
		"b1":b1,
		"W2":W2,
		"b2":b2
	}
	return parameters

n_x,n_h,n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x,n_h,n_y)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


#GRADED FUNCTION:forward_propagation
def forward_propagation(X,parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	Z1 = np.dot(W1,X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2,A1) + b2
	A2 = sigmoid(Z2)

	assert(A2.shape == (1,X.shape[1]))
	
	cache = {
		"Z1":Z1,
		"A1":A1,
		"Z2":Z2,
		"A2":A2
	}

	return A2,cache

X_assess,parameters = forward_propagation_test_case()
A2,cache = forward_propagation(X_assess,parameters)

print( np.mean(cache["Z1"]),np.mean(cache["A1"]),np.mean(cache["Z2"]),np.mean(cache["A2"]) )

#GRADED FUNCTION:compute_cost
def compute_cost(A2,Y,parameters):
	m = Y.shape[1]

	cost = -np.dot(np.log(A2+1e-7),np.transpose(Y)) - np.dot(np.log(1-A2+1e-7),np.transpose(1-Y))
	#logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
	#cost = - np.sum(logprobs)

	cost = np.squeeze(cost)

	#assert(isinstance(cost,float))
	return cost/m

A2,Y_assess,parameters = compute_cost_test_case()
cost = compute_cost(A2,Y_assess,parameters)
print("cost = " + str(cost))

#GRADED FUNCTION:backward_propagation
def backward_propagation(parameters,cache,X,Y):
	m = X.shape[1]

	W1 = parameters["W1"]
	W2 = parameters["W2"]

	A2 = cache["A2"]

	dz2 = A2 - Y
	dw2 = np.dot(dz2,np.transpose(cache["A1"]) ) / m
	db2 = np.sum(dz2,axis=1,keepdims=True) / m

	dz1 = np.dot(np.transpose(W2),dz2) * (1-np.power(cache["A1"],2))
	dw1 = np.dot(dz1,np.transpose(X)) / m
	db1 = np.sum(dz1,axis=1,keepdims=True) / m

	grads = {
		"dW1":dw1,
		"db1":db1,
		"dW2":dw2,
		"db2":db2,
	}

	return grads

parameters,cache,X_access,Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))


#GRADED FUNCTION：update_parameters
def update_parameters(parameters,grads,learning_rate=1.2):
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	b1 = parameters["b1"]
	b2 = parameters["b2"]

	dw1 = grads["dW1"]
	db1 = grads["db1"]
	dw2 = grads["dW2"]
	db2 = grads["db2"]

	W1 = W1 - learning_rate*dw1
	b1 = b1 - learning_rate*db1
	W2 = W2 - learning_rate*dw2
	b2 = b2 - learning_rate*db2

	parameters = {
		"W1":W1,
		"b1":b1,
		"W2":W2,
		"b2":b2
	}

	return parameters

parameters,grads = update_parameters_test_case()
parameters = update_parameters(parameters,grads)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


#GRADED FUNCTION：nn_model
def nn_model(X,Y,n_h,num_iterations=10000,print_cost=True):
	np.random.seed(2)
	n_x = layer_sizes(X,Y)[0]
	n_y = layer_sizes(X,Y)[2]

	parameters = initialize_parameters(n_x,n_h,n_y)
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	for i in range(num_iterations):
		A2,cache = forward_propagation(X,parameters)
		cost = compute_cost(A2,Y,parameters)
		grads = backward_propagation(parameters,cache,X,Y)

		parameters = update_parameters(parameters,grads)

		if print_cost and i%1000 == 0:
			print("Cost after iteration %i:%f" %(i,cost))
	return parameters

X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

#GRADED FUNCTION:predict
def predict(parameters,X):
	A2,cache = forward_propagation(X,parameters)
	A2[A2>0.5] = 1
	A2[A2<0.5] = 0
	predicts = A2.copy()
	return predicts

parameters,X_assess = predict_test_case()

predictions = predict(parameters,X_access)
print("predictions mean = " + str(np.mean(predictions)))

#BUild a model with a n_h-dimensional hidden layer
parameters = nn_model(X,Y,n_h=10,num_iterations=10000,print_cost = True)

#Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters,x.T),X,Y)
plt.title("Decision Boundary for hidden later size" + str(4))

plt.show()

predictions = predict(parameters,X)
print("Accuracy: %d" % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + "%" )


#Run Datasets
def run_datasets(X,Y,num_iterations,print_cost):
	#About 2 minutes to run
	plt.figure()
	hidden_layer_size = [1,2,3,4,5,20,50]
	for i,n_h in enumerate(hidden_layer_size):
		plt.subplot(5,2,i+1)
		plt.title("Hidden Layer of size %d" %n_h)
		parameters = nn_model(X,Y,n_h,num_iterations=5000,print_cost = True)
		plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
		predictions = predict(parameters, X)
		accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
		print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
	plt.show()

#Performance on other datasets
noisy_circles,noisy_moons,blobs,gaussian_quantiles,no_structure = load_extra_datasets()

datasets = {
	"noisy_circles":noisy_circles,
	"noisy_moons":noisy_moons,
	"blobs":blobs,
	"gaussian_quantiles":gaussian_quantiles
}

dataset = "noisy_moons"

X,Y = datasets[dataset]
X,Y = X.T,Y.reshape(1,Y.shape[0])

#make blobs binary
if dataset == "blobs":
	Y = Y%2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
plt.show()

run_datasets(X,Y,num_iterations=5000,print_cost=True)