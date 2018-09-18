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
# Case 1. all strategies have no skill
M = matrix(rnorm(N*TT, mean=0, sd=1), ncol=N, nrow=TT)
# Case 2. flat return -- ST treasury notes
M = matrix(0.05, ncol=N, nrow=TT)
# Case 3. many zero returns -- event-driven strategies
x = rnorm(N*TT, mean=0.1, sd=0.5)
x[which(rpois(N*TT, 1) > 0)] <- 0
M = matrix(x, ncol=N, nrow=TT)
# Case 4. serially correlated returns
M = matrix(arima.sim(model=list(ar=0.2), n=N*TT), ncol=N, nrow=TT)
X11();plot(M[,1], type="l")
# Case 5. outliers
M = matrix(rnorm(N*TT, mean=0, sd=1), ncol=N, nrow=TT)
M[which(rpois(TT, 0.01) > 0), 1] <- 4.0
X11();plot(M[,1], type="l")



Ms = DivideMat(M, S)
res <- TrainValSplit(Ms)
res1 <- CalcLambda(res$Train, res$Val, eval.method="ave")
Lambda = res1$lambda
X11();hist(Lambda, breaks=30, main="Distribution of Lamda")
(pbo = PBO(Lambda))
[1] 0.4126984

res1 = CalcLambda(res$Train, res$Val, eval.method="SR")
Lambda = res1$lambda
X11();hist(Lambda, breaks=30, main="Distribution of Lamda")
(pbo = PBO(Lambda))

png("LambdaDistrib.png")
hist(Lambda, breaks=30, main="Distribution of Lamda")
dev.off()




