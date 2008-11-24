function [gamma, alpha, beta, loglik] = hmmFwdBack(initDist, transmat, obslik)
% INPUT:
% initDist(i) = Pr(Q(1) = i)
% transmat(i,j) = Pr(Q(t) = j | Q(t-1)=i)
% obslik(i,t) = Pr(Y(t)| Q(t)=i)  
%
% OUTPUT
% gamma(i,t) = p(Q(t)=i | y(1:T))
% alpha(i,t)  = p(Q(t)=i| y(1:t))
% beta(i,t) propto p(y(t+1:T) | Q(t=i))
% loglik = log p(y(1:T))

[alpha, loglik] = hmmFilter(initDist, transmat, obslik);
beta = hmmBackwards(transmat, obslik);
gamma = normalize(alpha .* beta, 1);% make each column sum to 1


