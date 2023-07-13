## Learning Mapping Rules for Ontology to Relational Alignment

This repository is part of the [BESS
project](https://sites.google.com/view/symbolicsupervision) which
explores interactions between machine learning and logic. In
particular, we consider scenarios where the learning supervision is
provided wholly or in part by logical constraints, also referred to as
"symbolic supervision".


As a first step in the project,  we explore Disjunctive Supervision (DS), which is strongly related to Partial Label Learning (PLL). In both setups, each input is associated with a set of outputs, instead of a single correct one, as in the classical case. The difference between PLL and DS is that the former assumes a single, unknown, correct output corrupted by noise, while any of the provided outputs are considered correct for DS. This work is presented in paper [Towards Unbiased Exploration in Partial Label Learning](https://arxiv.org/abs/2307.00465).

This repository contains code needed to reproduce experimental results related to rule learning. The codebase has two independent parts:

- There is an **Extraction** module, responsible for extracting disjunctive datasets from the [RODI benchmark](https://www.cs.ox.ac.uk/isg/tools/RODI/), which is described in [extract/README.md](extract/README.md). This section also describes how to extract the datasets used in the paper.

- There is a **Learning** module, responsible for training models on disjunctive datasets, which is described in [learn/README.md](learn/README.md). This section describes how to run particular experiments mentioned in the paper.

