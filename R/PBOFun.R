# CombSymCVFun.R -- functions to implement the Combinatorially Symmetric Cross Validation
# Sep 11, 2018
# (c) Horace W Tso
# Ref : Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2016).
#       The probability of backtest overfitting. https://www.carma.newcastle.edu.au/jon/backtest2.pdf
#
#       Lopez de Prado (2018), Advances in Financial Machine Learning, John Wiley & Sons, Inc.
#

#' Divide a matrix into N chunks of equal size
#'
#' @param M matrix with number of rows divisible by n
#' @param n number of submatrices to break M into
#' @description Split matrix M into equal chunks, where each chunk has the
#'   same number of columns and rows. For example, if the original matrix has
#'   10 columns and 100 rows, and n=5, this function generates five matrices of
#'   size 20 x 10. Function returns a list of length n.
#' @return list of length n, where each element is a matrix.
#' @author Horace W. Tso <horacetso@gmail.com>
#' @examples
#'    N = 20 # no of strategies
#'    TT = 1000 # no of observations
#'    S = 20 # no of partitions
#'    M = matrix(rnorm(N*TT, mean=0.1, sd=1), ncol=N, nrow=TT)
#'    Ms = DivideMat(M, S)
#'    length(Ms)
#' @export
DivideMat = function(M, n) {
  TT = nrow(M)
  m = floor(TT / n)
  ix = c(seq(from=1, to=TT, by=m), TT+1) # TT+1 is needed
  Ms = list()
  for ( i in 1:n )
    Ms[[i]] = M[ix[i]:(ix[i+1]-1),]
  return(Ms)
}


#' Form the train/validation split by generating all combinations of the given list of matrices
#' taking half of them at a time.
#'
#' @param Ms list of equal-size matrices
#' @description This function splits the data into a train set and a validation set as many times
#'    as the number of combinations of drawing n/2 from n partitions. The number of this combinations
#'    is given by Bin(n, n/2), or binomial coefficient of n items taking n/2 at a time. Each combination
#'    provides the indices to the list Ms, and vertical stacking of these individual submatrices forms
#'    a training matrix. What is not selected in a combination index set are used as the validation set.
#' @details If the order of the rows in the original matrix is chronological, then each of training and
#'    validation matrix respects this chronological order.
#'
#'    WARNING : The resulting lists grow exponentially with the number of partitions. For example,
#'    when N=10, there are 252 splits, so the length of the returning list is 504. Doubling N, the
#'    list would have 369,512 matrices.
#' @references
#'    Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2016). _The probability of backtest overfitting_.
#'    \url{https://www.carma.newcastle.edu.au/jon/backtest2.pdf}
#'
#'    Lopez de Prado (2018), _Advances in Financial Machine Learning_, John Wiley & Sons.
#' @return list of two lists : Train, Val, where each is a list of length n/2 of
#'   matrices from the given Ms. Train <==> J, Val <==> J_bar in Bailey et al.
#' @author Horace W. Tso <horacetso@gmail.com>
#' @seealso [PBO::CalcLambda()]
#' @examples
#'    N = 20 # no of strategies
#'    TT = 1000 # no of observations
#'    S = 20 # no of partitions
#'    M = matrix(rnorm(N*TT, mean=0.1, sd=1), ncol=N, nrow=TT)
#'    Ms = DivideMat(M, S)
#'    res <- TrainValSplit(Ms)
#'    length(res$Train)
#'    length(res$Val)
#'    head(res$Train[[1]])
#' @export
TrainValSplit = function(Ms) {
  require(gtools)
  n = length(Ms)
  # Out of n items, take (n/2) at a time, generate all unique combinations
  ix.mat = combinations(n, n/2) # matrix of size Bin(n, n/2) x (n/2)
  Ncomb = nrow(ix.mat)
  iset = 1:n
  Train.ll = list()
  Val.ll = list()
  for ( i in 1:Ncomb ) {
    A = NULL
    B = NULL
    ia = ix.mat[i,]
    ib = setdiff(iset, ia) # everything not in ia
    # NOTE : follow the time order of each submatrix
    for ( j in 1:length(ia) )
      A = rbind(A, Ms[[ia[j]]])
    for ( j in 1:length(ib) )
      B = rbind(B, Ms[[ib[j]]])
    Train.ll[[i]] = A
    Val.ll[[i]] = B
  }
  return(list(Train=Train.ll, Val=Val.ll))
}

#' Calculate lambda and other attributes of the combinatoric train and validation matrices
#'
#' @param Train list of matrices
#' @param Val list of matrices, same length as Train
#' @param eval.method method of evaluating the columns (strategies), default is 'ave', but support 'SR'
#' @param risk.free.rate risk free rate for Sharpe ratio calculation, default is 0.02.
#' @description Calculate the lambda as defined in step (g) on p.12 of Bailey et al (2015).
#'    Lambda is the logit of the relative rank (w.bar) of the best training strategy
#'    in the matching validation set J_bar. It is calculated as follow,
#'      For each train set,
#'      1) find the column in train set with the best performance (eg. highest Sharpe ratio),
#'      2) find its corresponding validation performance in the validation set Val,
#'      3) calculate the rank of this column in Val,
#'      4) lambda is the logit of this Val rank
#' @details The default method of evaluation is 'ave', which averages the columns. The best column
#'    is the one with the maximum average. This would be appropriate if the matrix contains returns over
#'    a fixed period. The alternative is 'SR', or the Sharpe ratio.
#'    If the data are returns over a time period, it is better to use Sharpe ratio ("SR").
#' @references
#'    Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2016). _The probability of backtest overfitting_.
#'    \url{https://www.carma.newcastle.edu.au/jon/backtest2.pdf}
#'
#'    Lopez de Prado (2018), _Advances in Financial Machine Learning_, John Wiley & Sons.
#' @return list of elements : lambda, n.star, w.bar, IS.best, OOS
#' @author Horace W. Tso <horacetso@gmail.com>
#' @seealso [PBO::PBO()]
#' @examples
#'    M = matrix(rnorm(N*TT, mean=0.1, sd=1), ncol=N, nrow=TT)
#'    Ms = DivideMat(M, S)
#'    res <- TrainValSplit(Ms)
#'    res1 <- CalcLambda(res$Train, res$Val, eval.method="ave")
#'    (Lambda = res1$lambda)
#' @export
CalcLambda = function(Train, Val, eval.method="ave", risk.free.rate=0.02, rank.ties.method="random", verbose=FALSE) {
  Ncomb = length(Train)
  N = ncol(Train[[1]])
  lambda = double(Ncomb)
  n.star = integer(Ncomb)
  w.bar = double(Ncomb)
  IS.best = double(Ncomb)
  OOS = double(Ncomb)
  Kendall = double(Ncomb)
  Spearman = double(Ncomb)
  for ( i in 1:Ncomb ) {
    cat("==== i : ", i, " ======\n")
    if ( eval.method == "ave") {
      R = colMeans(Train[[i]]) # metric is the average of a column
      R.bar = colMeans(Val[[i]])
    } else if ( eval.method == "SR" ) {
      ret.tr = colMeans(Train[[i]])
      ret.v = colMeans(Val[[i]])
      stdev.tr = apply(Train[[i]], 2, sd)
      stdev.v = apply(Val[[i]], 2, sd)
      R = ifelse( stdev.tr > 0, (ret.tr - risk.free.rate) / stdev.tr, 0)
      R.bar = ifelse(stdev.v > 0, (ret.v - risk.free.rate) / stdev.v, 0)
    } else {
      stop("i don't know what to do with this eval.method.")
    }
    #if (verbose) cat("R :", R, "\n")
    #if (verbose) cat("R.bar :", R.bar, "\n")
    r.max = max(R, na.rm=FALSE) # error if NA is found
    ix = which( R == r.max )
    cnt = length(ix)
    if ( cnt > 1 ) warning("more than one max in R; cnt : ", cnt, "\n")
    IS.best[i] = r.max
    # if there are more than one unique maximum, randomly pick one
    n.star[i] = ix[sample(length(ix), size=1)]
    if ( verbose ) cat("n* : ", n.star[i], "\t")
    OOS[i] = R.bar[n.star[i]]
    #if ( verbose ) cat("R.bar[n.star] : ", OOS[i], "\n")
    if ( verbose ) cat("rank R.bar[n*] : ", base::rank(R.bar, ties.method=rank.ties.method)[n.star[i]] , "\n")
    w.bar[i] = base::rank(R.bar, ties.method=rank.ties.method)[n.star[i]] / N
    lambda[i] = log(w.bar[i] / (1 - w.bar[i]))
    # Kendall and Spearman rank correlation
    Kendall[i] = cor(R, R.bar, method="kendall")
    Spearman[i] = cor(R, R.bar, method="spearman")
  }
  return(list(lambda=lambda, n.star=n.star, w.bar=w.bar, IS.best=IS.best, OOS=OOS, Kendall=Kendall, Spearman=Spearman))
}

#' Calculate the Probability of Backtest Overfitting (PBO)
#'
#' @param lambda numeric vector, which is the relative rank logit, or lambda from CalcLambda()
#' @description Calculate the Probability of Backtest Overfitting as in section 11.6
#'    of Lopez de Prado's book. It is the area of the lambda distribution to the
#'    left of 0, including 0. In other words, it is the cumulative probability that the out-of-sample
#'    relative rank of a strategy is lower (or worse) than the in-sample rank.
#'
#'         PBO = Integral_{-Inf}^{0} p(lambda) dlambda
#'
#' @references
#'    Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2016). _The probability of backtest overfitting_.
#'    \url{https://www.carma.newcastle.edu.au/jon/backtest2.pdf}
#'
#'    Lopez de Prado (2018), _Advances in Financial Machine Learning_, John Wiley & Sons.
#' @return numeric value, which is the probability of backtest overfitting
#' @author Horace W. Tso <horacetso@gmail.com>
#' @seealso [PBO::CalcLambda()]
#' @examples
#'    M = matrix(rnorm(N*TT, mean=0.1, sd=1), ncol=N, nrow=TT)
#'    Ms = DivideMat(M, S)
#'    res <- TrainValSplit(Ms)
#'    res1 <- CalcLambda(res$Train, res$Val, eval.method="ave")
#'    (pbo = PBO(res1$lambda))
#' @export
PBO = function(lambda) {
  return(sum( lambda <= 0 )/length(lambda))
}







