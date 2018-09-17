# PBO: Probability of Backtest Overfitting 

One of the most challenging problems in machine learning is the assessment of a model's ability to generalize to unseen data. For applications in finance, this is made particularly acute by the sequential nature of the data and the failure of many standard cross-validation techniques to deal with long range dependence. 

Any trading strategy built on overfitted models puts the firm's capital and the manager's reputation at risk. Thus, the field of financial machine learning is in great need of techniques to assess the generalization capability of learning algorithms. 

In Bailey et al[1], they take the approach that a model's performance should be consistent in any
subset of the training-validation pairs. By evaluating every combination of these pairs, one could 
quantify empirically the degree (probability) that a model trained on historic data is overfitted (backtest overfitting).

This is a R implementation of their algorithm, separately described in a recent book[2]. The variable names closely follow the notation in Chapter 11 of said book. 

## Installation
To install directly from github, open a terminal, type R, then

    devtools::install_github('htso/PBO')

## Dependencies
You need the following packages. To install from a terminal, type 

    install.packages("gtools", "foreach", "parallel")

## Tutorial

Let's start with a simple example and see how the various functions work.

The typical case is that you think you've found a neat algorithm to make step-ahead forecast after searching through a long history of stock prices. The training procedure that landed you the final model depends on a couple of hyperparameters. After some grid search, you found the optimal combination of these hyperparamters. (Don't worry about the multiple testing issue for now, although that's a major problem.) 

At each attempt to find a better model, you run through all the data you have, beginning in 1989-09-07 and ending in 2018-09-07, and generate a return for every week in the 30 year period, totally 1560 performance numbers. The grid search made 20 trials, meaning that you really have 20
different models trained with different hyperparamters. 

Now, put all of these into a matrix, which would have 1,560 rows and 20 columns, and call this `M`. For illustration, I use gaussian random variates to represent the return matrix `M`. 

    N = 20 
    TT = 1560 

    set.seed(99989)
    M = matrix(rnorm(N*TT, mean=0, sd=1), ncol=N, nrow=TT)

Pick an _even_ number `S`, which, as I'll explain below, should be more than 6 and less than 20.

    S = 10

Then, you want to divide `M` into 10 submatrices of equal size, which in this case would have 156 rows and 20 columns. The function `DivideMat()` does that for you.

    Ms = DivideMat(M, S)
    length(Ms)

where `Ms` is a list of length 10, whose elements are 156 x 20 matrices. Next, you generate all combinations of 10 objects taking 5 at a time, resulting in a total of 252 combinations. This is done by calling `TrainValSplit`,

    res <- TrainValSplit(Ms)  

The function returns an object `res` which is a list of two lists, each of length 252. The first is the combinatorially "perturbed" training sets, and the second has the data that's not in the first one, which is the validation set we want. If you find this confusing, pause and think about what this is doing. 

With these two (very long) list objects, you're ready to calculate `lambda`, which gives you a distribution from which you estimate the empirical probability of overfitting. 

    res1 <- CalcLambda(res$Train, res$Val, eval.method="ave")
    Lambda = res1$lambda

Lambda is a vector of length 252, ie. each value corresponds to one combination, which is the relative rank of the best in-sample strategy out-of-sample. Again, if this doesn't make sense, don't worry about it.

The last step is to call function `PBO`, which returns a value in range of [0,1), and this is interpreted as a probability, 

    pbo = PBO(Lambda)
    [1] 0.4126984

So, there is a 41% probability of overfitting. High PBO is a warning of overfitting, but it may not be sufficient evidence to reject a model. There are cautionary tales with regard to its proper usage, which I'll write about in a separate blog. Here is the histogram of Lambda.

![lambdadistrib](https://user-images.githubusercontent.com/5498043/45640714-3ac2a500-ba68-11e8-957e-fd61f08bf00d.png)

The choice of `S` is not critical but must be considered in the context of the computing capability you have access to. For exa[1] 0.4126984mple, on a Linux box with 40 Gb of RAM, I could push `S` to around 20. Anything beyond 20 does not seem practical -- Bin(22,11) is 705,432, which requires eight times more RAM. 

This concludes a short tutorial of PBO.

The above code can be found in the /demo subfolder. To run it, 

    setwd(system.file(package="PBO"))
    demo(Tutorial)

## WARNING
As in many other combinatoric problems, the scale of computation and memory requirement grow exponentially with the number of partitions. It is recommended that `S` be set to no more than 20. 

## Cousins
A python implementation of PBO is forthcoming. Stay tuned.

## Platforms
Developed and tested on Linux (ubuntu 14.04), R 3.4.4.

## Bugs
Please report all bugs to horacetso@gmail.com

## References
[1] Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2016). The probability of backtest overfitting. https://www.carma.newcastle.edu.au/jon/backtest2.pdf

[2] Lopez de Prado (2018), Advances in Financial Machine Learning, John Wiley & Sons, Inc.



