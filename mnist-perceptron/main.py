'''
Author: Daniel Schuster
This is a machine learning program to classify MNIST handwritten numbers, 0-9
using the perceptron learning algorithm. Generates a acc.csv file containing
the accuracy data for the test and training data after each epoch, and a
confusion_matrix.csv file containing the confusion matrix data.
Basic perceptron learning algorithm can achieve approximately 85% accuracy.
Uses the MNIST dataset in .csv format.
'''

import numpy
import random
import csv

#Must be updated to match the location and name of the test and training data files
TEST_FILE = "mnist_test.csv"
TRAIN_FILE = "mnist_train.csv"

NUM_TRAINING_DATA = 60000
NUM_TEST_DATA = 10000
NUM_FIELDS = 784
W_ROWS = 10
W_COLUMNS = 785
STEPSIZE_SMALL = 0.001
STEPSIZE_MED = 0.01
STEPSIZE_LARGE = 0.1
MAX_EPOCHS = 20
   

def main():
   random.seed(1)
   numpy.random.seed(1)
   training_acc = []
   test_acc = []
   epoch = 0
   acc_delta = 100
   stepsize = STEPSIZE_LARGE
   confusion_matrix = numpy.zeros((10, 10))

   training_matrix = load_training_data()
   test_matrix = load_test_data()
   weights = create_weight_matrix()
   
   training_acc.append(test_training_data(training_matrix, weights))
   test_acc.append(test(test_matrix, weights, confusion_matrix))
   print(f"after epoch {epoch}: training acc {training_acc[-1]:.4f}, test_accuracy {test_acc[-1]:.4f} (stepsize {stepsize})")
   epoch += 1

   if epoch > 2:
      acc_delta = training_acc[epoch] - training_acc[epoch - 1]
   while epoch <= MAX_EPOCHS and acc_delta > 0.01:
      training_acc.append(train(training_matrix, weights, stepsize))
      test_acc.append(test(test_matrix, weights, confusion_matrix))
      print(f"after epoch {epoch}: training acc {training_acc[-1]:.4f}, test_accuracy {test_acc[-1]:.4f} (stepsize {stepsize})")
      epoch += 1
   
   export_confusion_matrix(confusion_matrix)

   results = []
   results.append(training_acc)
   results.append(test_acc)
   file = open("acc.csv", "w+")
   write = csv.writer(file)
   write.writerows(results)


def export_confusion_matrix(confusion_matrix, filename="confusion_matrix.csv"):
   with open(filename, mode='w', newline='') as file:
      writer = csv.writer(file)

      #create header
      header = ["Actual \\ Predicted"] + [str(i) for i in range(10)]
      writer.writerow(header)

      #write data
      for actual, row in enumerate(confusion_matrix):
         writer.writerow([actual] + list(row))


def test(test_matrix, weights, confusion_matrix):
   num_success = 0
   for i in range(0, NUM_TEST_DATA):
      ground_truth = int(test_matrix[i][0])
      x = numpy.insert(test_matrix[i][1:], 0, 1)

      #compute perceptron outputs
      output = numpy.dot(weights, x)
      guess = numpy.argmax(output) 
      confusion_matrix[ground_truth][guess] += 1
      if guess == ground_truth:
         num_success += 1

   return num_success / NUM_TEST_DATA
   

def test_training_data(training_matrix, weights):
   num_success = 0
   for i in range(0, NUM_TEST_DATA):
      ground_truth = int(training_matrix[i][0])
      x = numpy.insert(training_matrix[i][1:], 0, 1)

      #compute perceptron outputs
      output = numpy.dot(weights, x)

      guess = numpy.argmax(output) 
      if guess == ground_truth:
         num_success += 1

   return num_success / NUM_TEST_DATA 


def train(training_matrix, weights, stepsize):
   num_success = 0
   for i in range(0, NUM_TRAINING_DATA):
      ground_truth = int(training_matrix[i][0])
      x = numpy.insert(training_matrix[i][1:], 0, 1) #input vector

      #compute perceptron outputs
      outputs = numpy.dot(weights, x)
      y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      
      for j in range(0, 10):
         if outputs[j] > 0:
            y[j] = 1 #perceptron fires

      guess = numpy.argmax(outputs) 
      if guess == ground_truth:
         num_success += 1
      else:
         for j in range(10):
            #Compute t(i) - y(i) (1 if correct, 0 or -1 otherwise)
            t = 1 if j == ground_truth else 0  #correct class
            y_j = y[j]  #predicted output for the j-th perceptron

            # Update weights for perceptron j if necessary
            delta = stepsize * (t - y_j) * x
            weights[j] += delta 

   return num_success / NUM_TRAINING_DATA 


def load_training_data():
   #open file, create matrix
   with open(TRAIN_FILE, 'r') as training_file:
      training_matrix = numpy.loadtxt(training_file, delimiter=',')

   #normalize data values (except the ground truth in first column)
   for i in range(0, NUM_TRAINING_DATA):
      for j in range(1, NUM_FIELDS + 1): 
         training_matrix[i][j] /= 255

   return training_matrix


def load_test_data():
   #open file, create matrix
   with open(TEST_FILE, 'r') as test_file:
      test_matrix = numpy.loadtxt(test_file, delimiter=',')

   #normalize data values (except the ground truth in first column)
   for i in range(0, NUM_TEST_DATA):
      for j in range(1, NUM_FIELDS + 1): 
         test_matrix[i][j] /= 255

   return test_matrix


def create_weight_matrix():
   weights = []
   for i in range(0, W_ROWS):
      row = []
      for j in range(0, W_COLUMNS):
         row.append(random.uniform(-0.05, 0.05))
      weights.append(row)

   weights = numpy.array(object=weights)
   return weights


if __name__ == "__main__":
   main()