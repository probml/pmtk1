function [precMat, covMat] = ggmIPFR(S, G, varargin)
% Uses R code to find MLE of gaussian graphical model given S=cov(data) and graph G

% For instructions on calling R from Matlab, see
% http://www.cs.ubc.ca/~mdunham/tutorial/external.html#21

%#author Giovanni Marchetti 
%#url http://cran.r-project.org/web/packages/ggm/index.html

[n] = process_options(varargin, 'sampleSize', 1);
d = size(G,1);

%{
  S <- matrix(c(3.0230,    1.2580,    1.0040,    1.2580,    1.7090,    0.8420,    1.0040,    0.8420,    1.1160), ncol=3)
  G <- matrix( c(0,    1,     0,     1,     0,     1,     0 ,    1,     0), ncol=3)
  d <- 3
  n <- 1
  dimnames(S) <- list(1:d, 1:d)
  dimnames(G) <- list(1:d, 1:d)
  stuff <- fitConGraph(G, S, n)
  covMat <- stuff$Shat
%}

openR;
G = setdiag(G,0);
evalR('library(ggm)');
putRdata('S',S);
putRdata('G',G);
putRdata('d',d);
putRdata('n',n);
evalR('stuff<-1') 
evalR('Shat<-1') 
evalR('dimnames(S) <- list(1:d, 1:d)')
evalR('dimnames(G) <- list(1:d, 1:d)')
evalR('stuff <- fitConGraph(G, S, n)') 
evalR('Shat <- stuff$Shat') % inverse covariance matrix
%evalR('Shat <- solve(S)')
covMat = getRdata('Shat');
precMat = inv(covMat);
closeR;
