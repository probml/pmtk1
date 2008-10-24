function rnorm = rMNorm(m,V,T)
% rMNorm generates a qxT array  of normal draws
%  from the p-dim N(m,V)
%
p=length(m); 
rnorm = repmat(reshape(m,p,1),1,T) +chol(V)'*randn(p,T);



