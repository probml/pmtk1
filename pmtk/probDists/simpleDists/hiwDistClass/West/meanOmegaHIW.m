function [Om] = meanOmegaHIW(G,bG,DG) 
% posterior mean of Omega on decomposable graph G when its inverse Sigma~HIW(bG,DG)
% input:  G - the usual cell array whose 2 cells are the comps and seps

C = G{1}; S = G{2}; k=size(S,2);  p=size(DG,1); Om=zeros(p);
j=C(1).ID;  Om(j,j)=(bG+C(1).dim-1)*inv(DG(j,j));  % first component 
for c=2:k,      
    % visit separator c
     j=S(c).ID; Om(j,j)=Om(j,j)-(bG+S(c).dim-1)*inv(DG(j,j)); 
    % visit component c
     j=C(c).ID; Om(j,j)=Om(j,j)+(bG+C(c).dim-1)*inv(DG(j,j)); 
end
%
