# two_sensor
The code provided here reproduces the analysis and figures in the manuscript "Multi-grip classification-based prosthesis control with two sensors". The manuscript is currently available on [biorxiv](https://www.biorxiv.org/content/10.1101/579367v1).

## Instructions
The recommended way of reproducing the results is by using [Anaconda](https://anaconda.org/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and the provided environment file (`conda_env.yml`).

1. Clone the repository or download as zip.
2. Navigate in the repo you have just cloned/downloaded and create a conda environment using the file `conda_env.yml`. The environment will be called `two_sensor_env`. It should take approximately 3 min to setup this environment on a standard machine. See [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html?highlight=environment) for more details on conda environments (not required).
```
conda env create -f conda_env.yml
```

3. Activate the environment.

On Windows:
```
conda activate two_sensor_env
```
On Linux/OS X:
```
source activate two_sensor_env
```

4. To reproduce a figure in the manuscript navigate to the relevant directory (this is important, otherwise you will receive an error) and run the corresponding script. The data associated with each figure is located in the same directory. To reproduce the outcomes of the statistical results reported in the manuscript run the corresponding script. A report will be printed on the console. For example, assuming that you are inside the directory `offline_analysis`, you can run the following commands:
```
python Figure_3.py
python statistical_comparisons.py
```
Statistical comparison and other summaries will be printed on the console and figures will be saved in `.pdf` format in the same directory. 

## Instructions without using Anaconda/Miniconda
If you do not wish to use Anaconda/Miniconda, you will need a working Python 3.6/3.7 installation with the following packages (numbers in brackets indicate tested versions):
* [Numpy](http://www.numpy.org/) (1.16.2)
* [Scipy](https://www.scipy.org/) (1.2.1)
* [Pandas](https://pandas.pydata.org/) (0.24.2)
* [Statsmodels](https://www.statsmodels.org/stable/index.html) (0.9.0)
* [Matplotlib](https://matplotlib.org/) (3.0.3)
* [Seaborn](https://seaborn.pydata.org/) (0.9.0)

Once you have setup a working environment, follow steps 1 and 4 above.

## Script execution time
All scripts should take less than a minute to run.

## Contents
The following list provides details on the contents of each sub-directory and how to reproduce every figure in the manuscript.
* `offline_analysis`: Offline analysis. Reproduce Figure 3 and related statistical comparisons.
* `working_principle`: Working principle of the real-time control framework. Reproduce Figure 4.
* `real_time_analysis`: Analysis of results from real-time control experiment. Reproduce Figure 5, related statistical comparisons, and reported performance summaries.
* `emg_power`: Analysis of task practice on EMG power. Reproduce Figure 6.
* `metrics`: Offline and real-time performance metrics comparison. Reproduce Figure 7.
* `confidence_rejection`: Confidence-based rejection. Reproduce Figure 8.

## Issues/Feedback
If you run into any issues when trying to run the scripts or have any feedback on the code and/or results please open a new issue.
