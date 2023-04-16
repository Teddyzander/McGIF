## McGIF: Model Calibration for Group and Individual Fairness via Preferential Randomisation

![graphical abstract drawio (1)](https://user-images.githubusercontent.com/49641102/232171673-076d60ad-b951-4962-8bb6-51dd32772f52.svg)

### Command to run

*python max_equalised_odds.py --group [int] --grid [int] --show [bool] --file_name [str]*

### Arguments

1. ```group``` defines which class you are optimising. From the paper, 0=white, 1=black, 2=Hispanic, 3=Asian;
2. ```grid``` defines the density search space for p_a between each set of thresholds, e.g., --grid 11 gives p_a=[0, 0.1, 0.2, ..., 1];
3. ```show``` defines whether or not to show plots during the routine; and
4. ```file_name``` defines the prefix for each file to save the calculated thresholds and probability for each curve.


### Areas of interest:

- ```max_equalised_odds.py``` is the main file, and runs the script for finding continuous solutions that satisfy equalised odds and individual fairness;
- ```fairness_optimiser.py``` contains the code which finds each curve;
- ```equations/prob_functions.py``` contains the closed form definitions of each curve; and
- ```plotter_full.ipynb``` contains all the plotting routines.

