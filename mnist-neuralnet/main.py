'''
Author: Daniel Schuster
This program implements a neural network with one 
hidden layer to classify the MNIST dataset.
Running the program creates .csv files with data for the accuracy
over epochs and a confusion matrix of the classification vs actual.
Contains adjustable constants: the number of nodes in the hidden layer
is especially impactful.  With 100 hidden nodes, approximately 97%
accuracy is achieved, but training is slower than with fewer nodes.
'''

import numpy
import random
import csv

#path to directory where my mnist_train.csv and mnist_test.csv files are located.
#this must be updated if the files are in a different directory.
TRAIN_PATH = "mnist_train.csv" 
TEST_PATH = "mnist_test.csv" 

#MNIST dataset info
NUM_TRAINING_DATA = 60000
NUM_TEST_DATA = 10000
NUM_FIELDS = 784

#options for momentum value
MOMENTUM_ZERO = 0.00
MOMENTUM_SMALL = 0.25
MOMENTUM_MED = 0.50
MOMENTUM_LARGE = 0.90

#options for number of hidden nodes in network
HIDDEN_UNITS_SMALL = 20
HIDDEN_UNITS_MED = 50
HIDDEN_UNITS_LARGE = 100

W_COLUMNS = NUM_FIELDS + 1 #columns in W1 matrix (+1 for bias)
NUM_OUTPUT = 10 #number of output nodes (1 for each digit 0-9)

CONFUSION_FILENAME = "confusion_matrix.csv"
ACCURACY_FILENAME = "accuracy.csv"

#globals for adjustable hyperparameters
NUM_EPOCHS = 50
stepsize = 0.1
num_hidden = HIDDEN_UNITS_MED
momentum = MOMENTUM_SMALL

   
def main():
   random.seed(2)
   numpy.random.seed(2)
   training_acc = []
   test_acc = []
   epoch = 0
   confusion_matrix = numpy.zeros((10, 10))

   #create matrices for inputs and weights
   training_matrix = load_training_data()
   test_matrix = load_test_data()
   W1 = numpy.random.uniform(-0.05, 0.05, (num_hidden, W_COLUMNS)) #weights from input to hidden nodes
   W2 = numpy.random.uniform(-0.05, 0.05, (NUM_OUTPUT, num_hidden)) #weights from hidden to output nodes
   
   #perform an initial test on both training and test sets before any training is performed
   training_acc.append(test(training_matrix, NUM_TRAINING_DATA, W1, W2))
   test_acc.append(test(test_matrix, NUM_TEST_DATA, W1, W2))
   print(f"after epoch {epoch}: training acc {training_acc[-1]:.4f}, test_accuracy {test_acc[-1]:.4f}")
   epoch += 1

   #perform training and test accuracy for the specified number of epochs
   while epoch <= NUM_EPOCHS:
      training_acc.append(train(training_matrix, W1, W2))
      test_acc.append(test(test_matrix, NUM_TEST_DATA, W1, W2))
      print(f"after epoch {epoch}: training acc {training_acc[-1]:.4f}, test_accuracy {test_acc[-1]:.4f}")
      epoch += 1

   #final test after training is complete to generate confusion matrix
   test(test_matrix, NUM_TEST_DATA, W1, W2, confusion_matrix)
   
   #create .csv files with accuracy and confusion matrix results
   export_confusion_matrix(confusion_matrix)
   export_accuracy(training_acc, test_acc)


def export_accuracy(training_acc, test_acc):
   '''
   creates a single object containing both training and test accuracy data,
   then exports the data to acc.csv file that will be created in
   the PATH directory
   '''
   results = []
   results.append(training_acc)
   results.append(test_acc)
   file = open(ACCURACY_FILENAME, "w+")
   write = csv.writer(file)
   write.writerows(results)


def calc_output_error(o, t):
   '''
   calculates the error values for the output nodes
   return: vector of the error deltas for output nodes
   '''
   return o * (1 - o) * (t - o)

def calc_hidden_error(output_errors, h, W2):
   '''
   calculates the error values for the hidden nodes
   return: vector of the error deltas for hidden nodes
   '''
   #I was trying to get this to work with matrix operations instead of loops
   #but couldn't get it done in time
   delta = numpy.zeros(num_hidden)
   for j in range(num_hidden):
      delta[j] = h[j] * (1 - h[j]) * sum(W2[k][j] * output_errors[k] for k in range(NUM_OUTPUT))
   return delta

def export_confusion_matrix(confusion_matrix):
   '''
   exports the confusion matrix data as a .csv file
   that will be created in the PATH directory
   '''
   with open(CONFUSION_FILENAME, mode='w', newline='') as file:
      writer = csv.writer(file)

      #create header
      header = ["Actual \\ Predicted"] + [str(i) for i in range(10)]
      writer.writerow(header)

      #write data
      for actual, row in enumerate(confusion_matrix):
         writer.writerow([actual] + list(row))


def test(data_matrix, num_data, W1, W2, C=[]):
   '''
   generates classifications for all the items in data_matrix.
   tracks number of correct classifications and returns the accuracy.
   optional argument for confusion matrix C to track classifications.
   return: classification accuracy, float 0-1
   '''
   num_success = 0

   for input_item in range(0, num_data):
      ground_truth = int(data_matrix[input_item][0])
      x = numpy.insert(data_matrix[input_item][1:], 0, 1) #x is input vector
      t = numpy.full((NUM_OUTPUT,), fill_value=0.1, dtype=float)
      t[ground_truth] = 0.9

      #calculate outputs of hidden nodes (h vector) and output nodes (o vector)
      h = activation_function(numpy.dot(W1, x))
      o = activation_function(numpy.dot(W2, h))
      guess = numpy.argmax(o)
      
      if guess == ground_truth:
         num_success += 1

      if len(C) > 0: #update confusion matrix if one was given as argument
         C[ground_truth][guess] += 1

   return num_success / num_data


def train(training_matrix, W1, W2):
   '''
   performs one full epoch of training on all data items in training_matrix.  Uses a
   batch size of 1.  Momentum and number of hidden nodes adjustable via global constants.
   Uses forward propogation to calculate output class, then backpropagation to update
   weights.
   return: training data classification accuracy of the current epoch
   '''
   W1_prev = numpy.zeros((num_hidden, W_COLUMNS))
   W2_prev = numpy.zeros((NUM_OUTPUT, num_hidden))
   num_success = 0

   for input_item in range(0, NUM_TRAINING_DATA):
      ground_truth = int(training_matrix[input_item][0])
      x = numpy.insert(training_matrix[input_item][1:], 0, 1) #x is input vector
      t = numpy.full((NUM_OUTPUT,), fill_value=0.1, dtype=float)
      t[ground_truth] = 0.9

      #calculate outputs of hidden nodes (h vector) and output nodes (o vector)
      h = activation_function(numpy.dot(W1, x))
      o = activation_function(numpy.dot(W2, h))
      guess = numpy.argmax(o)
      
      if guess == ground_truth:
         num_success += 1

      #calculate errors and use backpropagation to update weights
      output_errors = calc_output_error(o, t)
      hidden_errors = calc_hidden_error(output_errors, h, W1)
      W1_prev = update_W1(W1, hidden_errors, x, W1_prev)
      W2_prev = update_W2(W2, output_errors, h, W2_prev)

   #return training accuracy for the current epoch
   return num_success / NUM_TRAINING_DATA 


def update_W1(W1, hidden_errors, x, W1_prev_delta):
   '''
   updates the weights from input to hidden nodes (W1 matrix) using backpropagation.
   return: matrix of the weight deltas to be used as W1_prev_delta next iteration
   '''
   delta_W1 = stepsize * numpy.outer(hidden_errors, x) + momentum * W1_prev_delta
   W1 += delta_W1
   return delta_W1


def update_W2(W2, output_errors, h, W2_prev_delta):
   '''
   updates the weights from hidden to output nodes (W2 matrix) using backpropagation.
   return: matrix of the weight deltas to be used as W2_prev_delta next iteration
   '''
   delta_W2 = stepsize * numpy.outer(output_errors, h) + momentum * W2_prev_delta
   W2 += delta_W2
   return delta_W2


def activation_function(z):
   '''
   applies sigmoid function to the scalar input z, and returns the result
   '''
   return 1 / (1 + numpy.exp(-z))


def load_training_data():
   '''
   opens mnist_train.csv from the directory designated by PATH constant, then
   creates a numpy matrix representation of the data and normalizes the values
   to between 0 and 1, with exception of first index that contains the ground truth value.
   return: normalized training data matrix with ground truth values in the first column
   '''
   with open(TRAIN_PATH, 'r') as training_file:
      training_matrix = numpy.loadtxt(training_file, delimiter=',')

   training_matrix[:, 1:] /= 255
   return training_matrix


def load_test_data():
   '''
   opens mnist_test.csv from the directory designated by PATH constant, then
   creates a numpy matrix representation of the data and normalizes the values
   to between 0 and 1, with exception of first index that contains the ground truth value.
   return: normalized test data matrix with ground truth values in the first column
   '''
   with open(TEST_PATH, 'r') as test_file:
      test_matrix = numpy.loadtxt(test_file, delimiter=',')

   test_matrix[:, 1:] /= 255
   return test_matrix


if __name__ == "__main__":
   main()