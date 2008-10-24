function G = mkGraphSymmetric(G)
% Ensure G(i,j) and G(j,i) are both either zero or non zero
% If the graph is weighted, the weights can be different in each direction
% eg G = [0 2 1; 
%         1 0 0;
%         0 0 0],
% mkGraphSymmetric(G) returns
%         [0 2 1;
%          1 0 0;
%          1 0 0]


%  add edge a-b if a->b OR a<-b
%G = sign(triu(G)+triu(G)'+tril(G)+tril(G)');

[is,js] = find(G==0);
for k=1:length(is)
  i = is(k); j = js(k);
  if G(i,j)==0
    G(i,j) = G(j,i);
  end
end
G = setdiag(G,0);
