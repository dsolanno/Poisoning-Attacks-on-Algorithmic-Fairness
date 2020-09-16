# Poisoning-Attacks-on-Algorithmic-Fairness
Code and supplementary material for "Poisoning Attacks on Algorithmic Fairness", ECML 2020.

Paper is available [here](https://arxiv.org/abs/2004.07401)


## Introduction

This work introduces an optimization framework to
craft poisoning samples that against Algorithmic fairness, in particular, the implemented functions are prepared to create new samples that compromise the disparate impact metric and all the ones correlated with it. 

We perform experiments in two scenarios: 
* a “black-box” attack in which the attacker only has access to a set of data sampled from the same distribution as the original training data, but not the model nor the original training set, 
* and a “white-box” scenario in which the attacker has full access to both.

Code is based on a fork of [SecML](https://secml.github.io/), adapted with new target functions allowing optimize the attacks against 

## How to use and extend

The main code is contained in the folder *fairness* where there are three main sources:

* First, there is a Python script, used to validate that the implemented function effectivelly attacks the disparate impact metric in a toy scenario.

* Second, there is a Python notebook containing the code for the experiments done on the synthetic dataset explained in the paper

* Third, there is another notebook containing the experiments on a real-world dataset.


