## Introduction

This is a reference implementation for the following models:

* M. Guthrie, A Leblois, A. Garenne, T. Boraud, "Interaction between cognitive
  and motor cortico-basal ganglia loops during decision making: a computational
  study", Journal of Neurophysiology, 2013. [![Build Status](https://travis-ci.org/rougier/basal-ganglia.svg?branch=guthrie_et_al_2013)](https://travis-ci.org/rougier/basal-ganglia)

* C. Piron, D. Kase, M. Topalidou, M. Goillandeau, H. Orignac, T. N'Guyen,
  N.P. Rougier, T. Boraud, "The Globus Pallidus Pars Interna in Goal-Oriented
  and Routine Behaviors: Resolving a Long-Standing Paradox", Movement
  Disorders, 2016.  [![Build Status](https://travis-ci.org/rougier/basal-ganglia.svg?branch=piron_et_al_2016)](https://travis-ci.org/rougier/basal-ganglia)


## Installation

It requires python, numpy, cython and matplotlib:

```bash
$ pip install numpy
$ pip install cython
$ pip install matplotlib
$ pip install tqdm
```

To compile the model, just type (original instruction) :

```bash
$ python setup.py build_ext --inplace
```

If encountering (like me) problem loading numpy, rather type :
```bash
$ python3 setup.py install --user
```

Then you can run a single trial:

```bash
$ python single-trial.py
```


Or the full version:

```bash
$ python experiment-guthrie.py
```
