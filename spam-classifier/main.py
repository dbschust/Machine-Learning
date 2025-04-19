'''
Author: Daniel Schuster
This is a machine learning program to classify emails as spam vs non-spam
using Gaussian Naive Bayes classification.  Prints the accuracy, precision,
and recall of the results, and a confusion matrix of the classifications.
Uses the spambase dataset provided by the UC Irvine Machine Learning Repository.
Gaussian Naive Bayes achieves approximately 80% accuracy on the dataset. 
'''

import numpy as np
import pandas as pd
import math

#numerical constants
NUM_FEATURES = 57
MIN_SD = 0.0001
AVOID_LOG0 = 10 ** -50

#indexing constants
SPAM = 1
NONSPAM = 0
TP = 0
FP = 1
TN = 2
FN = 3

#PATH must lead to the parent directory containing the spambase.data
PATH = "spambase/"


def main():
   #put spambase data into numpy matrix with 4601 rows, 58 columns.
   #last column is the ground truth (1 for spam, 0 for nonspam) 
   data_matrix = load_spambase(f"{PATH}spambase.data")

   #split into training and test sets, calculate needed stats of training data set
   training_matrix, test_matrix = split_data(data_matrix)
   priors = calc_prior(training_matrix)
   means, sds = calc_stats(training_matrix)

   #classify the test data set, display results
   confusion_matrix = classify(test_matrix, priors, means, sds)
   display_results(confusion_matrix)


def display_results(confusion_matrix):
   '''
   calculates and prints the accuracy, precision, and recall based on 
   the confusion matrix data, prints the confusion matrix
   '''
   print(f"accuracy: {(confusion_matrix[TP] + confusion_matrix[TN]) / sum(confusion_matrix)}")
   print(f"precision: {confusion_matrix[TP] / (confusion_matrix[TP] + confusion_matrix[FP])}")
   print(f"recall: {confusion_matrix[TP] / (confusion_matrix[TP] + confusion_matrix[FN])}")

   print("\nConfusion Matrix:")
   print(f"               Predicted 0    Predicted 1")
   print(f"Actual 0 |  TN: {confusion_matrix[TN]:<5}       FP: {confusion_matrix[FP]:<5}")
   print(f"Actual 1 |  FN: {confusion_matrix[FN]:<5}       TP: {confusion_matrix[TP]:<5}")


def calc_prior(training_matrix):
   '''
   calculates the prior probabilities that an email in the training data set
   is spam vs nonspam.  returns a list containing both prior probabilities
   '''
   total = len(training_matrix)
   spam = 0
   for row in training_matrix:
      if row[-1] == 1:
         spam += 1

   prior_spam = spam / total
   prior_nonspam = 1 - prior_spam
   return [prior_nonspam, prior_spam]


def split_data(data_matrix):
   '''
   splits the spambase data matrix into two equal halves after shuffling the dataset.
   returns the newly created training and test matrices as a tuple
   '''
   middle = len(data_matrix) // 2
   np.random.shuffle(data_matrix)
   training_matrix = data_matrix[:middle]
   test_matrix = data_matrix[middle:]
   return training_matrix, test_matrix


def load_spambase(filepath):
   '''
   loads the spambase dataset into numpy matrix, and returns it
   '''
   df = pd.read_csv(filepath, header=None)
   return df.to_numpy()


def calc_stats(training_matrix):
   '''
   computes std dev and mean for both the spam and nonspam classes of training data.
   returns data as a tuple of lists, with means grouped as a list and sds grouped as a list
   '''
   training_spam = training_matrix[training_matrix[:, -1] == 1]
   training_nonspam = training_matrix[training_matrix[:, -1] == 0]
   
   mean_spam, sd_spam = calc_sd_mean(training_spam)
   mean_nonspam, sd_nonspam = calc_sd_mean(training_nonspam)

   return [mean_nonspam, mean_spam], [sd_nonspam, sd_spam]


def calc_sd_mean(matrix):
   '''
   does the calculations for the mean and sd of the data features in a matrix.
   assumes the number of features in the matrix is NUM_FEATURES constant
   returns the calculated mean and sd as a tuple
   '''
   mean = np.zeros(NUM_FEATURES)
   sd = np.zeros(NUM_FEATURES)

   for i in range(NUM_FEATURES):
      mean[i] = np.mean(matrix[:, i])
      sd[i] = max(np.std(matrix[:, i]), MIN_SD)
   #note: if std dev is zero, assign small value to avoid divide by zero

   return mean, sd 


def classify(test_matrix, priors, means, sds):
   '''
   function to classify the data in the test set using gaussian naive bayes classification.
   creates a confusion matrix that tracks the number of true
   positive, false positive, true negative, and false negative guesses made on the test data.
   returns the confusion matrix as a list, indexed by constants TP, FP, TN, FN
   '''
   confusion_matrix = [0, 0, 0, 0]
   for datum in range(len(test_matrix)):
      guess = NONSPAM

      #calculate the result for spam class
      result_spam = math.log(priors[SPAM])
      for i in range(NUM_FEATURES):
         term1 = 1 / (math.sqrt(2 * math.pi) * sds[SPAM][i])
         term2 = math.e ** -(((test_matrix[datum][i] - means[SPAM][i]) ** 2) / (2 * (sds[SPAM][i] ** 2)))
         result_spam += math.log((term1 * term2) + AVOID_LOG0)

      #calculate the result for nonspam class
      result_nonspam = math.log(priors[NONSPAM])
      for i in range(NUM_FEATURES):
         term1 = 1 / (math.sqrt(2 * math.pi) * sds[NONSPAM][i])
         term2 = math.e ** -(((test_matrix[datum][i] - means[NONSPAM][i]) ** 2) / (2 * (sds[NONSPAM][i] ** 2)))
         result_nonspam += math.log((term1 * term2) + AVOID_LOG0)

      #determine guess class and actual class
      if result_spam > result_nonspam:
         guess = SPAM
      actual = test_matrix[datum][-1]

      #update values for confusion matrix data
      if actual == SPAM and guess == SPAM:
            confusion_matrix[TP] += 1
      elif actual == SPAM and guess == NONSPAM:
            confusion_matrix[FN] += 1
      elif actual == NONSPAM and guess == NONSPAM:
            confusion_matrix[TN] += 1
      elif actual == NONSPAM and guess == SPAM:
            confusion_matrix[FP] += 1

   return confusion_matrix


if __name__ == "__main__":
   main()