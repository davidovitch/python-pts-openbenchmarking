A simple prototype to read/edit/download PTS test profiles from
[OpenBenchMarking.org](http://openbenchmarking.org/), see also issue
[146](phoronix-test-suite/phoronix-test-suite#146) on the
[PTS](https://github.com/phoronix-test-suite/phoronix-test-suite) issue tracker.

The XML test results are converted into a ```pandas.DataFrame``` table that can
be used for advanced filtering and selection strategies. For plotting the data
the ```matplotlib``` library is used.

For an illustrated example, see the jupyter
[notebook](https://github.com/davidovitch/python-pts-openbenchmarking/blob/master/workflow.ipynb)
in this repository. Github renders the notebook, but as an alternative you
can also use the notebook viewer
[here](http://nbviewer.jupyter.org/github/davidovitch/python-pts-openbenchmarking/blob/master/workflow.ipynb).
The view the example no Python installation is required.

