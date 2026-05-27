source("funcs.r")
source("funcs2.r")
library(mvtnorm)
library(mnormt)
library(Rlab)
library(gTests)
library(expm)
library(MASS)
library(normwhn.test)
library("MVN")


d = 100   #dimension
m = 100   #sample size
distr = "normal"   #distribution
choice = "Sig3"   #Covariance matrix for the model

B = 1000  #replications
BB = 500  #repeat times for Step 2 in Algo 1
L = 1  #estimate p-values L times

Ypm = Opm = YYpm = YDpm = matrix(0,B,2)
pvFisher = pvmvShapiro = pvMardia_s = pvMardia_k = pvMardia = pvEp = pvhz = pvRoyston = rep(0,B)
pFisher = rep(0,d)

for (ii in 1:B){
  X = gen(d,m,distr,choice)
  ATX = adapt.thres.cov(X)
  eigenv <- eigen(ATX, symmetric = TRUE)
  e.vec <- as.matrix(eigenv$vectors)
  sqrS <- e.vec %*% diag(sqrt(eigenv$values^(-1)), ncol = d) %*% t(e.vec)

  XX = X %*% sqrS
  XD = X %*% diag(diag(sqrS))
  temp = getp(X,L=L,BB=BB)
  temp2 = getp(XX,L=L,BB=BB)
  temp3 = getp(XD,L=L,BB=BB)
  Ypm[ii,] = temp$Yp
  YYpm[ii,] = temp2$Yp  #with sigma_x^{-1/2} plug in
  YDpm[ii,] = temp3$Yp   #with D_x^{-1/2} plug in
  Opm[ii,] = temp$Op
  
  #Modified SW
  pvmvShapiro[ii] = mvShapiro.Test.adapt.thres.mod(X)$p.value
  
  aMardia = mvn(X)$multivariateNormality
  bMardia = aMardia$"p value"
  pvMardia_s[ii] = as.numeric(paste(bMardia[1]))
  pvMardia_k[ii] = as.numeric(paste(bMardia[2]))
   
  ahz = mvn(X,mvnTest="hz")$multivariateNormality
  bhz = ahz$"p value"
  pvhz[ii] = as.numeric(paste(bhz))

  #Ep needs d<m
  if (d<m) {
  	aroyston = mvn(X,mvnTest="royston")$multivariateNormality
 	broyston = aroyston$"p value"
  	pvRoyston[ii] = as.numeric(paste(broyston))
   
  	aep=normality.test2(X)
  	pvEp[ii] = aep
  } 
  
  #Fisher's method
  for (j in 1:d){
  	pFisher[j] = shapiro.test(XX[,j])$p.value
  }
  Fisher = -2*sum(log(as.numeric(pFisher)))
  pvFisher[ii] = 1-pchisq(Fisher, 2*d)
  
  
  print(c(ii, Ypm[ii,], YYpm[ii,], YDpm[ii,], Opm[ii,], pvmvShapiro[ii], pvFisher[ii]))
}

d
m
distr
choice
aY = getpow(Ypm)  # our test
aO = getpow(Opm)  # eFR
aYY = getpow(YYpm)  # our test with sigma_x^{-1/2} plug in
aYD = getpow(YDpm)  # our test with D_x^{-1/2} plug in

Mardia_s = getpow(as.matrix(pvMardia_s))  #skewness
Mardia_k = getpow(as.matrix(pvMardia_k))  #kurtosis
Mardia_Bonf = length(which(pvMardia_s<0.025 | pvMardia_k<0.025))/B  #Bonferrni
hz = getpow(as.matrix(pvhz))  #HZ
Royston = getpow(as.matrix(pvRoyston))  #Royston
Ep = getpow(as.matrix(pvEp))  #Ep

mvShapiro = getpow(as.matrix(pvmvShapiro))
Fisher.test = getpow(as.matrix(pvFisher))

aY
aO
aYY
aYD

Mardia_s
Mardia_k
Mardia_Bonf
hz
Royston
Ep
mvShapiro
Fisher.test

#save(Ypm,YYpm,YDpm,Opm,pvmvShapiro,pvFisher, file=paste(distr,choice,d,m,B,".RData", sep="_"))

