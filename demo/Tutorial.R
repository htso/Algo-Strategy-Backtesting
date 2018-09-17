# Tutorial.R -- tutorial on PBO algorithm (Bailey et al 2016)
# Sep 11, 2018
# (c) Horace W Tso

library(PBO)

home = "/where/you/put/this/package"
setwd(home)

# M is the matrix of returns.
# Each row is one period, while each column is a strategy (or strategy configuration)

N = 20 # no of strategies
TT = 1560 # no of observations
S = 10 # no of partitions

set.seed(99989)
# strategies have no skill
M = matrix(rnorm(N*TT, mean=0, sd=1), ncol=N, nrow=TT)
# flat return
M = matrix(0.1, ncol=N, nrow=TT)
# many returns are zero
x = rnorm(N*TT, mean=0.1, sd=0.5)
x[which(rpois(N*TT, 1) > 0)] <- 0
M = matrix(x, ncol=N, nrow=TT)



Ms = DivideMat(M, S)
res <- TrainValSplit(Ms)
res1 <- CalcLambda(res$Train, res$Val, eval.method="ave")
Lambda = res1$lambda
(pbo = PBO(Lambda))

res1 = CalcLambda(res$Train, res$Val, eval.method="SR")
Lambda = res1$lambda
(pbo = PBO(Lambda))







