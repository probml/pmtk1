function [sepsize, seps]=separators_cell(cliques, jtree)
% NOTE: if you just want a 1 x num_seps array of the sequence separators, use
%       seps_residuals_histories.m
% inputs: 1. cliques, a 1 x t cell array of the t nonempty cliques of g in
%           RIP ordering (from chordal_to_ripcliques_cell.m)
%         2. jtree, the associated t x t adjacency matrix of the junction tree
% output: 1. sepsize, a matrix array of the size of the separator sets, in which
%           sepsize(i,j) = [number of elements in intersection between
%           cliques cliques{i} and cliques{j} if they are adjacent, and zero else.
%         2. seps, a (num_cliques)x(num_cliques) cell array
%           in which seps{i,j}=cliques{i} intersect cliques{j}.

%#author Helen Armstrong
%#url http://www.library.unsw.edu.au/~thesis/adt-NUN/uploads/approved/adt-NUN20060901.134349/public/01front.pdf 

t=size(cliques,2);
sepsize=zeros(size(jtree)); % =num_cliques x num_cliques
seps=cell(size(jtree));
for i=1:t;
  for j=i+1:t;
    if jtree(i,j)==1;
      seps{i,j}=intersect(cliques{i}, cliques{j});
      sepsize(i,j)=length(intersect(cliques{i}, cliques{j}));
      % this version does the full matrix/cell array
      % which is symmetric so actually unnecessary.
      % But could be dangerous not to compute in case the wrong ordering
      % j, i, for j > i is used by one of the calling programs)
      seps{j,i}=seps{i,j};
      sepsize(j,i)=sepsize(i,j);
    end;
  end;
end;