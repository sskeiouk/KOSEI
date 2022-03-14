from __future__ import division
from random import seed, shuffle
import random
import math
import os
from collections import defaultdict
from sklearn import svm
import os,sys
import urllib2
sys.path.insert(0, './fair_classification/') # the code for fair classification is in this directory
import numpy as np
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints
import random
import time
from scipy.optimize import basinhopping
import config_census
import config_statlog
import config_bank
import copy
from sklearn.externals import joblib

random.seed(time.time())
start_time = time.time()

# Setting for each Dataset

if (sys.argv[3] == "Census"):
    params = config_census.params
    sensitive_param = config_census.sensitive_param
    perturbation_unit = config_census.perturbation_unit
    threshold = config_census.threshold
    input_bounds = config_census.input_bounds
    if (sys.argv[2] == "DT"):
        classifier_name = 'census/Decision_tree_standard_unfair.pkl'
    elif (sys.argv[2] == "MLPC"):
        classifier_name = 'census/MLPC_standard_unfair.pkl'
    elif (sys.argv[2] == "RF"):
        classifier_name = 'census/Random_Forest_standard_unfair.pkl'

elif (sys.argv[3] == "German"):
    params = config_statlog.params
    sensitive_param = config_statlog.sensitive_param
    perturbation_unit = config_statlog.perturbation_unit
    threshold = config_statlog.threshold
    input_bounds = config_statlog.input_bounds
    if (sys.argv[2] == "DT"):
        classifier_name = 'statlog/DecisionTree_statlog01.pkl'
    elif (sys.argv[2] == "MLPC"):
        classifier_name = 'statlog/MLPC_statlog01.pkl'
    elif (sys.argv[2] == "RF"):
        classifier_name = 'statlog/RFC_statlog01.pkl'

elif (sys.argv[3] == "Bank"):
    params = config_bank.params
    sensitive_param = config_bank.sensitive_param
    perturbation_unit = config_bank.perturbation_unit
    threshold = config_bank.threshold
    input_bounds = config_bank.input_bounds
    if (sys.argv[2] == "DT"):
        classifier_name = 'bank/DecisionTree_bank.pkl'
    elif (sys.argv[2] == "MLPC"):
        classifier_name = 'bank/MLPC_bank.pkl'
    elif (sys.argv[2] == "RF"):
        classifier_name = 'bank/RFC_bank.pkl'

# Aequitas algorithm

if (sys.argv[1] == "aequitas"):
    init_prob = 0.5
    direction_probability = [init_prob] * params
    direction_probability_change_size = 0.001

    param_probability = [1.0 / params] * params
    param_probability_change_size = 0.001

    name = 'sex'
    cov = 0

    global_disc_inputs = set()
    global_disc_inputs_list = []

    local_disc_inputs = set()
    local_disc_inputs_list = []

    tot_inputs = set()

    global_iteration_limit = 2000
    local_iteration_limit = 2000
    model = joblib.load(classifier_name)


    def normalise_probability():
        probability_sum = 0.0
        for prob in param_probability:
            probability_sum = probability_sum + prob

        for i in range(params):
            param_probability[i] = float(param_probability[i]) / float(probability_sum)


    class Local_Perturbation(object):

        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            s = self.stepsize
            param_choice = np.random.choice(xrange(params), p=param_probability)
            act = [-1, 1]
            direction_choice = np.random.choice(act, p=[direction_probability[param_choice],
                                                        (1 - direction_probability[param_choice])])

            if (x[param_choice] == input_bounds[param_choice][0]) or (x[param_choice] == input_bounds[param_choice][1]):
                direction_choice = np.random.choice(act)

            x[param_choice] = x[param_choice] + (direction_choice * perturbation_unit)

            x[param_choice] = max(input_bounds[param_choice][0], x[param_choice])
            x[param_choice] = min(input_bounds[param_choice][1], x[param_choice])

            ei = evaluate_input(x)

            if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
                direction_probability[param_choice] = min(
                    direction_probability[param_choice] + (direction_probability_change_size * perturbation_unit), 1)

            elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
                direction_probability[param_choice] = max(
                    direction_probability[param_choice] - (direction_probability_change_size * perturbation_unit), 0)

            if ei:
                param_probability[param_choice] = param_probability[param_choice] + param_probability_change_size
                normalise_probability()
            else:
                param_probability[param_choice] = max(param_probability[param_choice] - param_probability_change_size,
                                                      0)
                normalise_probability()

            return x


    class Global_Discovery(object):
        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            s = self.stepsize
            for i in xrange(params):
                random.seed(time.time())
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

            x[sensitive_param - 1] = 0
            return x


    def evaluate_input(inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[sensitive_param - 1] = 0
        inp1[sensitive_param - 1] = 1

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out0 = model.predict(inp0)
        out1 = model.predict(inp1)

        # return (abs(out0 - out1) > threshold)
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)


    def evaluate_global(inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[sensitive_param - 1] = 0
        inp1[sensitive_param - 1] = 1

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out0 = model.predict(inp0)
        out1 = model.predict(inp1)
        tot_inputs.add(tuple(map(tuple, inp0)))

        if (abs(out0 - out1) > threshold and tuple(map(tuple, inp0)) not in global_disc_inputs):
            global_disc_inputs.add(tuple(map(tuple, inp0)))
            global_disc_inputs_list.append(inp0.tolist()[0])

        # return not abs(out0 - out1) > threshold
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)


    def evaluate_local(inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[sensitive_param - 1] = 0
        inp1[sensitive_param - 1] = 1

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out0 = model.predict(inp0)
        out1 = model.predict(inp1)

        tot_inputs.add(tuple(map(tuple, inp0)))

        if (abs(out0 - out1) > threshold and (tuple(map(tuple, inp0)) not in global_disc_inputs)
                and (tuple(map(tuple, inp0)) not in local_disc_inputs)):
            local_disc_inputs.add(tuple(map(tuple, inp0)))
            local_disc_inputs_list.append(inp0.tolist()[0])

        # return not abs(out0 - out1) > threshold
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)

    print "Search started"
    starting_time = time.time()
    if (sys.argv[3] == "Census"):
        initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
    elif (sys.argv[3] == "German"):
        initial_input = [1, 4, 1, 23, 4, 4, 0, 1, 2, 20, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    elif (sys.argv[3] == "Bank"):
        initial_input = [1, 5, 1, 2, 0, 23, 0, 1, 2, 20, 5, 35, 39, 45, 30, 2]
    minimizer = {"method": "L-BFGS-B"}

    global_discovery = Global_Discovery()
    local_perturbation = Local_Perturbation()

    basinhopping(evaluate_global, initial_input, stepsize=1.0, take_step=global_discovery, minimizer_kwargs=minimizer,
                 niter=global_iteration_limit)

    for inp in global_disc_inputs_list:
        basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
                     niter=local_iteration_limit)

    print "Total evaluated data: " + str(len(tot_inputs))
    print "Number of discriminatory data: " + str(len(global_disc_inputs_list) + len(local_disc_inputs_list))
    print "Percentage of discriminatory data: " + str(float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100)
    elapsed_time = time.time() - starting_time
    print ("Execution_time: {0}".format(elapsed_time) + "[sec]")
    print "Search ended"

# KOSEI algorithm

if (sys.argv[1] == "kosei"):
    name = 'sex'
    cov = 0

    disc_inputs = set()
    disc_inputs_list = []

    tot_inputs = set()

    global_iteration_limit = 2000
    local_iteration_limit = 0
    local_cnt = 0

    model = joblib.load(classifier_name)


    class Global_Discovery(object):
        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            s = self.stepsize
            for i in xrange(params):
                random.seed(time.time())
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

            x[sensitive_param - 1] = 0
            return x


    def evaluate_disc(inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[sensitive_param - 1] = 0
        inp1[sensitive_param - 1] = 1

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out0 = model.predict(inp0)
        out1 = model.predict(inp1)
        tot_inputs.add(tuple(map(tuple, inp0)))

        if (abs(out0 - out1) > threshold and tuple(map(tuple, inp0)) not in disc_inputs):
            disc_inputs.add(tuple(map(tuple, inp0)))
            disc_inputs_list.append(inp0.tolist()[0])

        # return not abs(out0 - out1) > threshold
        # for binary classification, we have found that the
        # following optimization function gives better results
        return abs(out1 + out0)

    # Local search algorithm for each dataset

    def my_local_search_Census(inp):
        for param in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]:
            for direction in [-1, 1]:
                inp2 = copy.copy(inp)
                inp2[param] = inp2[param] + direction
                if (inp2[param] < input_bounds[param][0] and direction == -1):
                    continue
                elif (inp2[param] > input_bounds[param][1] and direction == 1):
                    continue
                elif (tuple(map(tuple, np.reshape(np.asarray(inp2), (1, -1)))) in tot_inputs):
                    continue
                evaluate_disc(inp2)
                global local_cnt
                local_cnt += 1


    def my_local_search_German(inp):
        for param in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
            for direction in [-1, 1]:
                inp2 = copy.copy(inp)
                inp2[param] = inp2[param] + direction
                if (inp2[param] < input_bounds[param][0] and direction == -1):
                    continue
                elif (inp2[param] > input_bounds[param][1] and direction == 1):
                    continue
                elif (tuple(map(tuple, np.reshape(np.asarray(inp2), (1, -1)))) in tot_inputs):
                    continue
                evaluate_disc(inp2)
                global local_cnt
                local_cnt += 1


    def my_local_search_Bank(inp):
        for param in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            for direction in [-1, 1]:
                inp2 = copy.copy(inp)
                inp2[param] = inp2[param] + direction
                if (inp2[param] < input_bounds[param][0] and direction == -1):
                    continue
                elif (inp2[param] > input_bounds[param][1] and direction == 1):
                    continue
                elif (tuple(map(tuple, np.reshape(np.asarray(inp2), (1, -1)))) in tot_inputs):
                    continue
                evaluate_disc(inp2)
                global local_cnt
                local_cnt += 1


    print "Search started"
    starting_time = time.time()
    if (sys.argv[3] == "Census"):
        initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
    elif (sys.argv[3] == "German"):
        initial_input = [1, 4, 1, 23, 4, 4, 0, 1, 2, 20, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    elif (sys.argv[3] == "Bank"):
        initial_input = [0, 5, 1, 2, 0, 23, 0, 1, 2, 20, 5, 35, 39, 45, 30, 2]
    minimizer = {"method": "L-BFGS-B"}

    global_discovery = Global_Discovery()

    basinhopping(evaluate_disc, initial_input, stepsize=1.0, take_step=global_discovery, minimizer_kwargs=minimizer,
                 niter=global_iteration_limit)

    local_iteration_limit = len(disc_inputs_list) * 2000

    for inp in disc_inputs_list:
        if (local_cnt < local_iteration_limit):
            if (sys.argv[3] == "Census"):
                my_local_search_Census(inp)
            elif (sys.argv[3] == "German"):
                my_local_search_German(inp)
            elif (sys.argv[3] == "Bank"):
                my_local_search_Bank(inp)
        else:
            break

    print "Total evaluated data: " + str(len(tot_inputs))
    print "Number of discriminatory data: " + str(len(disc_inputs_list))
    print "Percentage of discriminatory data: " + str(float(len(disc_inputs_list)) / float(len(tot_inputs)) * 100)
    elapsed_time = time.time() - starting_time
    print ("Execution_time:{0}".format(elapsed_time) + "[sec]")
    print "Search ended"