#{r r_yy function}
library(mvtnorm)

adapt.thres.cov = function(X, kk=5){
 #from authors
 m = dim(X)[1]
 d = dim(X)[2]
 e = exp(1)
 I = diag(1,d)
 Xbar = apply(X,2,mean)
 xcov0 = cov(X)

 sm = m*(kk-1)/kk
 lambda = rep(0,50)
 dif = rep(0,50)
 for (j in 1:50){
   lambda[j] = 6*j/50*sqrt(log(d))
   for (fi in 1:kk){
     m1 = m/kk*(fi-1)+1
     m2 = m/kk*fi
     SX = X[-(m1:m2),]
     SXbar = apply(SX,2,mean)
     Sxcov = cov(SX)
     cc1 = (t(SX^2))%*%(SX^2)/sm
     cc2 = ((t(SX))%*%(SX)/sm)*Sxcov
     cc3 = Sxcov^2
     t1 = cc1-2*cc2+cc3
     AA = sqrt(sm*(Sxcov^2)/(t1))

     CX = X[m1:m2,]
     Cxcov = cov(CX)
     dif[j] = dif[j] + norm(Sxcov*(abs(AA)>=lambda[j])-Cxcov,"F")^2

   }
 }
 thr = max(lambda[which(dif==min(dif))])
 cc1 = (t(X^2))%*%(X^2)/m
 cc1 = (t(X^2))%*%(X^2)/m
 cc2 = ((t(X))%*%(X)/m)*xcov0
 cc3 = xcov0^2
 t1 = cc1-2*cc2+cc3
 AA = sqrt(m*(xcov0^2)/(t1))
 Xcov = xcov0*(abs(AA)>=thr)
 eO=eigen (Xcov)
  de = max(-min (eO$values),0)+0.05
  Xcov = (Xcov + de*I)/(1+de)
 return(Xcov)
}








library(mvShapiroTest)


#Here is a version using the authors adapt.thres. covariance estimate with `mvShapiro.Test`.

#{r adapt.thres SW}
mvShapiro.Test.adapt.thres.mod=function (X) {
   dname <- deparse(substitute(X))
   if (is.vector(X) == TRUE) 
       X = cbind(X)
   stopifnot(is.matrix(X))
   n <- nrow(X)
   if (n < 12 || n > 5000) 
       stop("Sample size must be between 12 and 5000.")
   p <- ncol(X)
       x <- scale(X, scale = FALSE)
       eigenv <- eigen(adapt.thres.cov(X), symmetric = TRUE)
       e.vec <- as.matrix(eigenv$vectors)
       sqrS <- e.vec %*% diag(sqrt(eigenv$values^(-1)), ncol = p) %*% 
           t(e.vec)      ##this has been modified, originally there is no inverse, it calcuated Sigma^{1/2}, not Sigma^{-1/2}
       z <- t(sqrS %*% t(x))
       w <- rep(NA, p)
       for (k in 1:p) {
           w[k] <- shapiro.test(z[, k])$statistic
       }
       wast <- mean(w)
       y <- log(n)
       w1 <- log(1 - wast)
       m <- -1.5861 - 0.31082 * y - 0.083751 * y^2 + 0.0038915 * 
           y^3
       s <- exp(-0.4803 - 0.082676 * y + 0.0030302 * y^2)
       s2 <- s^2
       sigma2 <- log((p - 1 + exp(s2))/p)
       mu1 <- m + s2/2 - sigma2/2
       p.value <- pnorm(w1, mean = mu1, sd = sqrt(sigma2), lower.tail = FALSE)
       results <- list(statistic = c(MVW = wast), p.value = p.value, 
           method = "Generalized Shapiro-Wilk test for Multivariate Normality by Villasenor-Alva and Gonzalez-Estrada modified with Adaptive Threshold  covariance matrix estimate from Cai and Liu 2011", 
           data.name = dname)
       class(results) = "htest"
       return(results)

}


