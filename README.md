# KOSEI

We present KOSEI.

#Prerequisites

* Python 2.7.18
* numpy 1.14.5
* scipy 1.1.0
* scikit-learn 0.19.0

The authors used Pycharm CE 2021.2.1 as the development IDE.

# Scikit-Learn Classifiers
* Adult (Census Income): Scikit-Learn classifiers in the "census" directory
* Statlog (German Credit): Scikit-Learn classifiers in the "statlog" directory
* Bank Marketing: Scikit-Learn classifiers in the "bank" directory

# Config 
* Adult (Census Income): config_census.py
* Statlog (German Credit): config_bank.py
* Bank Marketing: config_statlog.py

The config files has the following data (same as the Udeshi's codes):
* params : The number of parameters in the data
* sensitive_param: The parameter under test.
* input_bounds: The bounds of each parameter
* classifier_name: Pickled scikit-learn classifier under test (only applicable to the sklearn files)
* threshold: Discrimination threshold.
* perturbation_unit: By what unit would the user like to perturb the input in the local search.
* retraining_inputs: Inputs to be used for the retraining.

# Demo
You can conduct experiments of Aequitas and KOSEI with KOSEI.py.
`python KOSEI.py <algorithm> <classifier> <dataset>`

* algorithm : aequitas or kosei
* classifier : DT or MLPC or RF
* dataset : Census or German or Bank

eg1. python KOSEI.py kosei DT German
eg2. python KOSEI.py aequitas DT Census

# iteration
We set the default iteration limits in the following:
global_iteration_limit = 2000 
local_iteration_limit = 2000


# Contact
* About KOSEI approach
  * Please contact sanoshin@doi.ics.keio.ac.jp for any comments/questions


