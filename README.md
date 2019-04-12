# two_sensor
The code provided here reproduces the analysis and figures in the manuscript "Multi-grip classification-based prosthesis control with two sensors". The manuscript is currently available on [biorxiv](https://www.biorxiv.org/content/10.1101/579367v1).

## Instructions
The recommended way of reproducing the results is by using [Anaconda](https://anaconda.org/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and the provided environment file (`conda_env.yml`).

Alternatively, you will need a Python 3.6/3.7 installation with the following packages (numbers in brackets indicate tested versions):
* [Numpy](http://www.numpy.org/) (1.16.2)
* [Scipy](https://www.scipy.org/) (1.2.1)
* [Pandas](https://pandas.pydata.org/) (0.24.2)
* [Statsmodels](https://www.statsmodels.org/stable/index.html) (0.9.0)
* [Matplotlib](https://matplotlib.org/) (3.0.3)
* [Seaborn](https://seaborn.pydata.org/) (0.9.0)

1. Clone the repository or download as zip.
2. Create conda environment from file (see [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html?highlight=environment) for detailed instructions on conda environments).
```
conda env create -f conda_env.yml
```

3. To reproduce a figure in the manuscript navigate to the relevant directory (this is important, otherwise you will receive an error) and run the corresponding script. The data associated with each figure is located in the same directory. To reproduce the outcomes of the statistical results reported in the manuscript run the corresponding script. A report will be printed on the console. For example, assuming that you are inside the directory `Figure_3`, you can run the following commands:
```
python Figure_3.py
python statistical_comparisons.py
```

## Issues/Feedback
If you run into any issues when trying to run the scripts or if have any feedback on the code and/or results please open a new issue and I will be happy to help or discuss.
