classdef tree < undirectedGraph
  
 
 methods
   function obj = tree(adjMat)
     % A tree is an undirected graph with no loops
     if nargin == 0
       return;
     end
     obj.adjMat = adjMat;
     if ~checkAcyclic(obj)
       warning('BLT:tree', 'not a tree!')
     end
   end
  
 end
  
end