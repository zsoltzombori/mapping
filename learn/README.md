## Learning Module

The **Learning** module contains all python code related to experiments. The models are trained using Tensorflow 2 and the [environment.yml](environment.yml) file describes the required python environment.

The main entry point to run an experiment is `train.py` which accepts a large number of command line parameters. For each of our experiments, we create a bash script that lists all the parameters and provides a reference to the "parent experiment" -- the one with which the results should be compared with -- as well as a comment describing the changes to the parent. All of our experiment files can be found in the `experiments` folder.

To run an experiment, it suffices to run the corresponding script, e.g.

`bash experiments/exp128.sh`

The configuration files specify the location of the dataset in the `basedir` parameter. All the datasets used in our final experiments are provided in the `data_compressed` folder. Note that the datasets have to be unpacked before running an experiment. Alternatively, you can regenerate the datasets, as described in the **Extraction** module in [../extract/README.md](../extract/README.md).

### Commands to run CMT experiments (Section 5.1, Table 1):

- prp-loss: `bash experiments/exp91.sh`
- nll-loss: `bash experiments/exp129.sh`

### Commands to run NPD experiments (Section 5.2, Table 2):

- prp-loss, uniform sampling: `bash experiments/exp120.sh`
- nll-loss, uniform sampling: `bash experiments/exp121.sh`
- prp-loss, realistic sampling: `bash experiments/exp127.sh`
- nll-loss, realistic sampling: `bash experiments/exp128.sh`

### Code used to generate Figures 1, 2, 3, 4, 5, 6:

- `python mlp.py`