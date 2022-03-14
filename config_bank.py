# ORIGINAL FILE
params = 16

sensitive_param = 1 # Starts at 1.

input_bounds = []

# INPUT SET for BANK dataset
#'''
input_bounds.append([0, 1]) # Discriminatory parameter
input_bounds.append([0, 11])
input_bounds.append([0, 2])
input_bounds.append([0, 3])
input_bounds.append([0, 1])
input_bounds.append([-9, 102])
input_bounds.append([0, 1])
input_bounds.append([0, 1])
input_bounds.append([0, 2])
input_bounds.append([1, 31])
input_bounds.append([0, 11])
input_bounds.append([0, 49])
input_bounds.append([1, 63])
input_bounds.append([-1, 87])
input_bounds.append([0, 58])
input_bounds.append([0, 3])

threshold = 0

perturbation_unit = 1

#retraining_inputs = "Retrain_Example_File.txt"