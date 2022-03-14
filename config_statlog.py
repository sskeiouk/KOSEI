# ORIGINAL FILE
params = 24

sensitive_param = 7 # Starts at 1.

input_bounds = []

# INPUT SET NUMBER 1: Original dataset, 5 values for the discriminatory parameter
'''
input_bounds.append([1, 4])
input_bounds.append([4, 72])
input_bounds.append([0, 4])
input_bounds.append([2, 184])
input_bounds.append([1, 5])
input_bounds.append([1, 5])
input_bounds.append([1, 4])
input_bounds.append([1, 4])
input_bounds.append([1, 4])
input_bounds.append([19, 75])
input_bounds.append([1, 3])
input_bounds.append([1, 4])
input_bounds.append([1, 2])
input_bounds.append([1, 2])
input_bounds.append([1, 2])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([1, 2])
#'''


# INPUT SET NUMBER 2: Modified dataset, 2 values for the discriminatory parameter
#'''
input_bounds.append([1, 4])
input_bounds.append([4, 72])
input_bounds.append([0, 4])
input_bounds.append([2, 184])
input_bounds.append([1, 5])
input_bounds.append([1, 5])
input_bounds.append([0, 1]) # Discriminatory parameter
input_bounds.append([1, 4])
input_bounds.append([1, 4])
input_bounds.append([19, 75])
input_bounds.append([1, 3])
input_bounds.append([1, 4])
input_bounds.append([1, 2])
input_bounds.append([1, 2])
input_bounds.append([1, 2])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([1, 2])
#'''

classifier_name = 'statlog/DecisionTree_statlog01.pkl'
#classifier_name = 'statlog/MLPC_statlog01.pkl'
#classifier_name = 'statlog/RFC_statlog01.pkl'

# Original config file
# INPUT SET NUMBER 1: Original
'''
input_bounds.append([1, 9])
input_bounds.append([0, 7])
input_bounds.append([0, 39])
input_bounds.append([0, 15])
input_bounds.append([0, 6])
input_bounds.append([0, 13])
input_bounds.append([0, 5])
input_bounds.append([0, 4])
input_bounds.append([0, 1])
input_bounds.append([0, 99])
input_bounds.append([0, 39])
input_bounds.append([0, 99])
input_bounds.append([0, 39])
#'''


threshold = 0

perturbation_unit = 1

#retraining_inputs = "Retrain_Example_File.txt"