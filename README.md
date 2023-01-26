# McGIF
Model Calibration for Individual and Group Fairness via Preferential Randomisation

Current state of the code is disgusting at best. It takes a while to run (sorry).

The file you want to run is (probably) max_equalised_odds.py

There are some settings at the beginning (lines 24-37)

* 'A' changes which class you are running the analysis on (see data folder for more detail)
* 'grid' refers to the search density for an optimal solution. 'grid=101' means that 101 probabilities are checked for each combination of thresholds.
* 'brute' is whether or not to brute force the optimal solution. Turning this off limits the threshold settings.
* 'delta' is how far of a range should we check that individual fairness is satisfied for
* 'epsilon' is the maximum change allowed from each point +- delta in the solution space
* 'force' is whether or not to check for individual fairness constraints at all (parameterised by delta and epsilon)
* Then there are some plotting flags and file name stuff.
