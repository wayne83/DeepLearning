import numpy as np
import time 
import h5py
import matplotlib.pyplot as plt 
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_x_orig,train_y,test_x_orig,test_y,classes = load_data()

'''
index = 10
plt.imshow(train_x_orig[index])
print("y = " + str(train_y[0,index]) + ".It's a " + classes[train_y[0,index]].decode("utf-8") + "picture.")
plt.show()
'''

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

'''
print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))
'''

#Reshape the training and test examples
train_x_flattern = train_x_orig.reshape(train_x_orig.shape[0],-1).T 
test_x_flattern = test_x_orig.reshape(test_x_orig.shape[0],-1).T

#Standardize data to have 
train_x = train_x_flattern / 255
test_x = test_x_flattern / 255
'''
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
'''

#Two_layer neural network
n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x,n_h,n_y)

def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=True):
	np.random.randn(1)
	grads = {}
	costs = []
	m = X.shape[1]

	(n_x,n_h,n_y) = layers_dims
	parameters = initialize_parameters(n_x,n_h,n_y)
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	for i in range(num_iterations):
		A1,cache1 = linear_activation_forward(X,W1,b1,"relu")
		A2,cache2 = linear_activation_forward(A1,W2,b2,"sigmoid")

		cost = compute_cost(A2,Y)
		costs.append(cost)
		if print_cost == True and i%100 == 0:
			print("Cost after iteration %i:%f" %(i,cost))

		dA2 = -(Y/A2) + (1-Y)/(1-A2)
		grads["dA1"],grads["dW2"],grads["db2"] = linear_activation_backward(dA2,cache2,"sigmoid")
		grads["dA0"],grads["dW1"],grads["db1"] = linear_activation_backward(grads["dA1"],cache1,"relu")

		parameters = update_parameters(parameters,grads,learning_rate)

		W1 = parameters["W1"]
		b1 = parameters["b1"]
		W2 = parameters["W2"]
		b2 = parameters["b2"]

	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()

	return parameters

'''
parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
predictions_train = predict(train_x,train_y,parameters)
predictions_test = predict(test_x,test_y,parameters)
'''


#L_layer_model
layers_dims = [12288,20,7,5,1]

def L_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=True):

	parameters = initialize_parameters_deep(layers_dims)
	costs = []

	for i in range(num_iterations):
		AL,caches = L_model_forward(X,parameters)
		cost = compute_cost(AL,Y)
		costs.append(cost)
		grads = L_model_backward(AL,Y,caches)
		parameters = update_parameters(parameters,grads,learning_rate)

		if print_cost == True and i%100 == 0:
			print("Cost after iteration %i:%f" %(i,cost))

	plt.plot(np.squeeze(costs))
	plt.ylabel("cost")
	plt.xlabel("iterations(per tens)")
	plt.title("Learning rate = " + str(learning_rate))
	plt.show()

	return parameters

#Train Start
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
pred_train = predict(train_x,train_y,parameters)
pred_test = predict(test_x,test_y,parameters)

print_mislabeled_images(classes,test_x,test_y,pred_test)
plt.show()


#Test Own Image
my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [1]         # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image) 
plt.show()
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")