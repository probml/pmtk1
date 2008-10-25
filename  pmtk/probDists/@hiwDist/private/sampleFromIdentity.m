function [Sigma, K] = sampleFromIdentity(G, delta)
% Draw random matrix Sigma ~ HIW(G, delta, I), K = inv(Sigma)
% G is a decomposableGraph, delta is a scalar

% Kevin Murphy, UBC June 2008
% Based on code by Helen Armstrong, UNSW 2005

perfect_order = G.perfectElimOrder;
rev_perf = perfect_order(end:-1:1);
g = G.adjMat;
p = size(g,1);
g_rev_perf = g(rev_perf,rev_perf);
% Diagonal elements are sqrt(chi-squared(delta))
Psi = zeros(p,p);
for i=1:p
  if i==p
    num_parents = 0;
  else
    % num parents wrt perfect order
    num_parents = sum(g_rev_perf(i,i+1:p));
  end
  Psi(i,i)= (chi2rnd(delta+num_parents))^(.5);
end
% Psi(i,j), j >i is N(0,1) if an edge exists, otherwise 
for i=1:p-1
  for j=i+1:p
    if (g_rev_perf(i,j) > 0)
      Psi(i,j)=randn;
    end
  end
end
K_rev_perf = Psi' * Psi;
sigma_id_rev_perf = inv(K_rev_perf);

% Now convert to original order
inverse_permute=zeros(1,p);
for j=1:p
  inverse_permute(j)=find(rev_perf==j);
end
K = K_rev_perf(inverse_permute, inverse_permute);
Sigma = sigma_id_rev_perf(inverse_permute, inverse_permute);
