function [alpha, loglik] = hmmForwards(initDist, transmat, obslik)
% INPUT:
% initDist(i) = Pr(Q(1) = i)
% transmat(i,j) = Pr(Q(t) = j | Q(t-1)=i)
% obslik(i,t) = Pr(Y(t)| Q(t)=i)  
%
% OUTPUT
% alpha(i,t)  = p(Q(t)=i| y(1:t))
% loglik = log p(y(1:T))

[K T] = size(obslik);
alpha = zeros(K,T);
[alpha(:,1), scale(1)] = normalize(initDist(:) .* obslik(:,1));
for t=2:T
  [alpha(:,t), scale(t)] = normalize((transmat' * alpha(:,t-1)) .* obslik(:,t));
end
loglik = sum(log(scale+eps));
