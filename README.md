# two_sensor
The code provided here reproduces the results in the manuscript "Multi-grip classification-based prosthesis control with two sensors", which is currently accessible on [biorxiv](https://www.biorxiv.org/content/10.1101/579367v1).

## Instructions
The recommended way of reproducing the results is by using [Anaconda](https://anaconda.org/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and the provided environment file (`conda_env`).

Alternatively, you will need a Python 3.6/3.7 installation with the following packages (numbers in brackets indicate tested versions):
* [Numpy](http://www.numpy.org/) (1.16.2)
* [Scipy](https://www.scipy.org/) (1.2.1)
* [Pandas](https://pandas.pydata.org/) (0.24.2)
* [Matplotlib](https://matplotlib.org/) (3.0.3)
* [Seaborn](https://seaborn.pydata.org/) (0.9.0)

1. Clone the repository or download as zip.
2. Create conda environment from file (see [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html?highlight=environment) for detailed instructions on conda environments).
```
conda env create -f conda_env.yml
```

3. Navigate to the directory relevant to the figure of the manuscript you wish to reproduce (e.g. Figure_3) and then run corresponding script (e.g. `python Figure_3.py`)
