# The learning module

This module contains all python code related to experiments and figure generation. The models are trained using Tensorflow~2.4 and the [environment.yml](environment.yml) file describes the required python environment.

The main entry point to run an experiment is `train.py` which accepts a large number of command line parameters. For each of our experiments, we create a bash script that lists all the parameters and provides a reference to the "parent experiment" -- the one with which the results should be compared with -- as well as a comment describing the changes to the parent. All of our experiment files can be found in the [experiments/](experiments/) folder.

To run an experiment, it suffices to run the corresponding script, e.g.

`bash experiments/exp128.sh`

The configuration files specify the location of the dataset in the `basedir` parameter. All the datasets used in our final experiments are provided in the [data_compressed/](data_compressed/) folder. Note that the datasets have to unpacked before running an experiment. These datasets were generated from the CMT and NPD challenges of the RODI benchmark, using the provided code.

