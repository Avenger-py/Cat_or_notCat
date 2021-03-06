import numpy as np
import scipy
from matplotlib import pyplot as plt
import h5py
from PIL import Image
from scipy import ndimage
from dataset_file import load_dataset
import scipy.misc


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
index = 7


train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T


train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


def sig(z):
    s = np.exp(z)/(1 + np.exp(z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, (float, int)))
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sig(np.dot(w.T, X) + b)
    cost = (-1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

    dz = A - Y
    dw = (1/m)*(np.dot(X, dz.T))
    db = (1/m)*(np.sum(dz))

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw, "db": db}
    return grads, cost


def opt(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i%100 == 0:
            costs.append(cost)
        if print_cost and i%100 == 0:
            print("Cost after %dth iteration: %f" %(i, cost))

        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sig(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        y_prediction[:, i] = (A[:, i] > 0.5) * 1

    assert(y_prediction.shape == (1, m))

    return y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w,b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = opt(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    y_prediction_train = predict(w, b, X_train)
    y_prediction_test = predict(w, b, X_test)
    print('Train accuracy: {}'.format(100 - np.mean(np.abs(y_prediction_train - Y_train))*100))
    print('Test accuracy: {}'.format(100 - np.mean(np.abs(y_prediction_test - Y_test))*100))

    d = {'costs': costs,
         'w': w,
         'b': b,
         'Y_prediction_train': y_prediction_train,
         'Y_prediction_test': y_prediction_test,
         'learning_rate': learning_rate,
         'num_iterations': num_iterations}

    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 4500, learning_rate = 0.005, print_cost = True)



img_name = ""
my_image = "test_images/" + img_name + ".jpg"   # change this to the name of your image file
fname = my_image
image = np.array(ndimage.imread(fname, flatten=False))
image = image/255.
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
plt.imshow(image)
plt.show()
