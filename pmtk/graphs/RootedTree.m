classdef RootedTree <  DirectedGraph
  
  properties
    root;
    preorder; % parents before children
    postorder; % children before parents
    parent; % parent(i) is the unique parent of i on route to root
  end
  
 methods
   function obj = RootedTree(adjMat, root)
     % Adjmat should be the adjmat of a tree
     % All arrows point away from the root (root defaults to 1)
     if nargin == 0, return; end
     if nargin < 2, root = 1; end
     obj.adjMat = adjMat;
     n = length(adjMat);
     T = sparse(n,n); % not the same as T = sparse(n) !
     directed = 0;
     [d, obj.preorder, obj.postorder, hascycle, f, obj.parent] = dfs(obj, root, directed);
     if hascycle
       warning('PMTK:tree', 'not a tree!')
     end
     pred = obj.parent;
     for i=1:length(pred)
       if pred(i)>0
         T(pred(i),i)=1;
       end
     end
     obj.adjMat = T;
   end
  
 end
  
end