library(ade4) # mstree function

##-----------------------------------------
## Construct graph
##-----------------------------------------

getdis = function(y){
  n = dim(y)[1]
  G = y%*%t(y)
  g = diag(G)
  dis = sqrt( matrix(rep(g,n),n) + matrix(rep(g,n),n,byrow=T) - 2*G )
  dis
}


gen = function(d=20,m=100,distr="normal",choice="Sig1",chisq.df=10){
  I = diag(1,d)
  
  if (choice=="Sig1"){
    S1 = SS = I
  }else if (choice=="Sig11"){
    S1 = I
    for (i in 1:d){
    	S1[i,i] = runif(1,1,5)
    }
    SS = sqrtm(S1)
  }else if (choice=="Sig12"){
    S1 = I
    for (i in 1:d){
    	S1[i,i] = runif(1,1,20)
    }
    SS = sqrtm(S1)
  }else if (choice=="Sig2"){
    Omega2 = diag(1,d)
    for (i in 1:d){
      for (j in i:d){
        Omega2[i,j]=0.5^{abs(i-j)}
        Omega2[j,i]=Omega2[i,j]
      }
    }
    S1 = Omega2
    SS = sqrtm(S1)
  }else if (choice=="Sig3"){
    Sigma3 = diag(1,d)
    for (i in 1:(d-1)){
      for (j in (i+1):d){
        Sigma3[i,j] = rbern(1,0.02)*runif(1)
        Sigma3[j,i] = Sigma3[i,j]
      }
    }
    de = abs(min(eigen(Sigma3)$values))+0.05
    Sigma3 = (Sigma3 + de*I)/(1+de)
    S1 = Sigma3
    SS = sqrtm(S1)
  }
  
  mu = rep(0,d)
  if (distr=="normal"){
    X = rmvnorm(m, mu, S1)
  }else if (distr=="t05"){
      X = rmvt(m, S1, df=0.5*d)
  }else if (distr=="t025"){
      X = rmvt(m, S1, df=0.25*d)
  }else if (distr=="t2"){
      X = rmvt(m, S1, df=2*d)
  }else if (distr=="t4"){
      X = rmvt(m, S1, df=4*d)
  }else if (distr=="mixture"){
    ss1 = 1-1.8/sqrt(d)
    ss2 = 1+1.8/sqrt(d)
    X = rbind(rmvnorm(m/2,rep(0,d),ss1*diag(1,d)), rmvnorm(m/2,rep(0,d),ss2*diag(1,d)))%*%SS
  }else if (distr=="partial_normal_01"){
  	X = rmvnorm(m, mu, S1)
  	prop = 0.1*d
  	I1 = diag(1,prop)
  	X[,1:prop] = rmvt(m, I1, df=d/4)
  }else if (distr=="partial_normal_02"){
  	X = rmvnorm(m, mu, S1)
  	prop = 0.2*d
  	I1 = diag(1,prop)
  	X[,1:prop] = rmvt(m, I1, df=d/4)
  }else if (distr=="partial_normal_03"){
  	X = rmvnorm(m, mu, S1)
  	prop = 0.3*d
  	I1 = diag(1,prop)
  	X[,1:prop] = rmvt(m, I1, df=d/4)
  }else if (distr=="partial_normal_04"){
  	X = rmvnorm(m, mu, S1)
  	prop = 0.4*d
  	I1 = diag(1,prop)
  	X[,1:prop] = rmvt(m, I1, df=d/4)
  }else if (distr=="partial_normal_05"){
  	X = rmvnorm(m, mu, S1)
  	prop = 0.5*d
  	I1 = diag(1,prop)
  	X[,1:prop] = rmvt(m, I1, df=d/4)
  }else if (distr=="chisquare3"){
    chisq.df=3
    X0 = matrix(rchisq(m*d,chisq.df),m)-chisq.df
    X = X0%*%SS/sqrt(chisq.df*2)
  }else if (distr=="chisquare5"){
    chisq.df=5
    X0 = matrix(rchisq(m*d,chisq.df),m)-chisq.df
    X = X0%*%SS/sqrt(chisq.df*2)
  }else if (distr=="chisquare10"){
    chisq.df=10
    X0 = matrix(rchisq(m*d,chisq.df),m)-chisq.df
    X = X0%*%SS/sqrt(chisq.df*2)
  }else if (distr=="chisquare20"){
    chisq.df=20
    X0 = matrix(rchisq(m*d,chisq.df),m)-chisq.df
    X = X0%*%SS/sqrt(chisq.df*2)
  }
  X
}

get.XYstat = function(distM){
  N = dim(distM)[1]
  m = N/2
  nn = apply(distM,1,which.min)
  
  XX = length(which(nn[1:m]<=m))
  YY = length(which(nn[(m+1):N]>m))
  
  ## the number of possibilities that the two arrows share the endpoint
  temp = as.vector(table(nn))
  a = max(temp)
  share = 0
  if (a>1){
    for (i in 2:a){
      share = share + choose(i,2)*length(which(temp==i))
    }
  }
  
  ## number of mutual nearest neighbors * 2
  mutual = length(which((nn[nn]-1:N)==0))
  
  mu.XX = m*(m-1)/(N-1)
  V.XX = m^2*(m-1)^2/(N*(N-1)*(N-2)*(N-3))*(N+mutual+(m-2)/(m-1)*share*2 -2*N/(N-1))
  Cov.XY = m^2*(m-1)^2/(N*(N-1)*(N-2)*(N-3))*(mutual-3*N-2*share+(4*N-6)*N/(N-1))
  Sigma = matrix(c(V.XX,Cov.XY,Cov.XY,V.XX),2)
  
  R = c(XX-mu.XX, YY-mu.XX)
  SXY1 = sum(abs(R))
  SXY2 = R%*%ginv(Sigma)%*%R
  
  return(list(XX=XX, YY=YY, SXY1=SXY1, SXY2=SXY2))
}

getR = function(E, ids){
  R = 0
  for (i in 1:nrow(E)){
    e1 = is.na(match(E[i,1],ids))
    e2 = is.na(match(E[i,2],ids))
    if ((e1+e2)==1) R = R+1
  }
  R
}

getp = function(X, L, BB, kk=5, CX.generate=FALSE){
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
  
  ratio0v = Xratio0v = SXY1_0v = SXY2_0v = ori0v = S0v = rep(0,L)
  for (l in 1:L){
    Y = rmvnorm(m, Xbar, Xcov)
    XY = rbind(X,Y)
    mydist = getdis(XY)
    
    ## nn
    mydist2 = mydist
    diag(mydist2) = max(mydist)+100
    temp = get.XYstat(mydist2)
    ratio0v[l] = temp$YY/m
    Xratio0v[l] = temp$XX/m
    SXY1_0v[l] = temp$SXY1
    SXY2_0v[l] = temp$SXY2
    #nn = apply(mydist2,1,which.min)
    #ratio0 = length(which(nn[(m+1):(m*2)]>m))/m
    #Xratio0 = length(which(nn[1:m]<=m))/m
    
    E = mstree(as.dist(mydist))
    ori0v[l] = getR(E,1:m)
    temp = g.tests(E,1:m,(m+1):(2*m),"g")
    S0v[l] = temp$generalized$test.statistic
  }
  
  ratio0 = mean(ratio0v)
  Xratio0 = mean(Xratio0v)
  SXY1_0 = mean(SXY1_0v)
  SXY2_0 = mean(SXY2_0v)
  ori0 = mean(ori0v)
  S0 = mean(S0v)

  ratio = Xratio = SXY1 = SXY2 = ori = S = matrix(0,BB,L)
  
  if (CX.generate){
    CX=rmnorm(m*BB, Xbar, Xcov)
    Cxcov = cov(CX)
  }else{
    Cxcov = Xcov
  }
  
  for (j in 1:BB){
    YY = rmvnorm(m, Xbar, Xcov)
    YYbar = apply(YY,2,mean)
    yycov = cov(YY)
    
    cc1 = (t(YY^2))%*%(YY^2)/m
    cc2 = ((t(YY))%*%(YY)/m)*yycov
    cc3 = yycov^2
    t1 = cc1-2*cc2+cc3
    AA = sqrt(m*(yycov^2)/(t1))
    
    lambda = rep(0,50)
    dif = rep(0,50)
    for (jj in 1:50){
      lambda[jj] = 6*jj/50*sqrt(log(d))
      dif[jj] = norm(yycov*(abs(AA)>=lambda[jj])-Cxcov,"F")
    }
    thr = max(lambda[which(dif==min(dif))])
    YYcov = yycov*(abs(AA)>=thr)
    
    eO=eigen (YYcov)
    de = max(-min (eO$values),0)+0.05
    YYcov = (YYcov + de*I)/(1+de)
    
    for (l in 1:L){
      
      Z = rmvnorm(m,YYbar,YYcov)
      YZ = rbind(YY,Z)
      mydist.YZ = getdis(YZ)
      mydist2.YZ = mydist.YZ
      diag(mydist2.YZ) = max(mydist.YZ)+100
      temp = get.XYstat(mydist2.YZ)
      ratio[j,l] = temp$YY/m
      Xratio[j,l] = temp$XX/m
      SXY1[j,l] = temp$SXY1
      SXY2[j,l] = temp$SXY2
      
      E = mstree(as.dist(mydist.YZ))
      ori[j,l] = getR(E,1:m)
      temp = g.tests(E,1:m,(m+1):(2*m),"g")
      S[j,l] = temp$generalized$test.statistic
      
    }
  }
  
  ratio.L = apply(ratio,1,mean)
  Xratio.L = apply(Xratio,1,mean)
  SXY1.L = apply(SXY1,1,mean)
  SXY2.L = apply(SXY2,1,mean)
  ori.L = apply(ori,1,mean)
  S.L = apply(S,1,mean)
  
  Yp = c(length(which(abs(ratio.L-mean(ratio.L))>=abs(ratio0-mean(ratio.L))))/BB, length(which(abs(ratio[,1]-mean(ratio[,1]))>=abs(ratio0v[1]-mean(ratio[,1]))))/BB)
  Xp = c(length(which(abs(Xratio.L-mean(Xratio.L))>=abs(Xratio0-mean(Xratio.L))))/BB, length(which(abs(Xratio[,1]-mean(Xratio[,1]))>=abs(Xratio0v[1]-mean(Xratio[,1]))))/BB)
  Op = c(length(which(abs(ori.L-mean(ori.L))>=abs(ori0-mean(ori.L))))/BB, length(which(abs(ori[,1]-mean(ori[,1]))>=abs(ori0v[1]-mean(ori[,1]))))/BB)
  Sp = c(length(which(S.L>=S0))/BB, length(which(S[,1]>=S0v[1]))/BB)
  SXY1p = c(length(which(SXY1.L>=SXY1_0))/BB, length(which(SXY1[,1]>=SXY1_0v[1]))/BB)
  SXY2p = c(length(which(SXY2.L>=SXY2_0))/BB, length(which(SXY2[,1]>=SXY2_0v[1]))/BB)

  return(list(Yp=Yp, Xp=Xp, Op=Op, Sp=Sp, SXY1p=SXY1p, SXY2p=SXY2p))
}

getpow = function(pm, alpha=0.05){
  B = dim(pm)[1]
  L = dim(pm)[2]
  r = rep(0,L)
  for (i in 1:L){
    r[i] = length(which(pm[,i]<alpha))
  }
  r/B
}






