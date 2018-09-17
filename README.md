# PBO: Estimate the probability of backtest overfitting 

One of the most challenging problems in machine learning is the assessment of a model's ability to generalize to unseen data. For applications in finance, this is made particularly acute by the sequential nature of the data and the failure of many standard cross-validation techniques to deal 
with long range dependence. 

Any trading strategy built on overfitted models puts the firm's capital and the manager's reputation at risk. Thus, the field of financial machine learning is in great need of techniques to assess 
the generalization capability of learning algorithms. 

In Bailey et al[1], they take the approach that a model's performance should be consistent in any
subset of the training-validation pairs. By evaluating every combination of these pairs, one could 
estimate the probability that a model trained on historic data is overfitted (backtest overfitting).

This is an R implementation of their algorithm, separately described in a recent book[2]. The variable names follows the notation in Chapter 11 of Prado's book. 

# Installation
To install directly from github, open a terminal, type R, then

    devtools::install_github('htso/PBO')

# Dependencies
You need the following packages. To install from a terminal, type 

    install.packages("gtools", "foreach", "parallel")

# Tutorial
I provide a tutorial in the /demo subfolder. To run it, 

    setwd(system.file(package="PBO"))
    demo(Tutorial)


# WARNING
As in many other combinatoric problems, the scale of computation and memory requirement grow exponentially with the number of partitions. It is recommended that S be set to no more than 20. 

# Platforms
Developed and tested on Linux (ubuntu 14.04), R 3.4.4.

# Bugs
Please report all bugs to horacetso@gmail.com

[1] Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2016). The probability of backtest overfitting. https://www.carma.newcastle.edu.au/jon/backtest2.pdf

[2] Lopez de Prado (2018), Advances in Financial Machine Learning, John Wiley & Sons, Inc.



