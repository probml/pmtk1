classdef Graph 
  
  properties
    adjMat;
    directed = false;
  end
  
 %methods(Abstract = true)
 %  e = nedges(obj);
 %  ns = neighbors(obj, v);
 %end

 %% Main methods
 methods
   function obj = Graph(adjMat)
     if nargin == 0
       obj.adjMat = [];
     else
       obj.adjMat = adjMat;
     end
   end
   
   function h=draw(obj)
       if obj.directed
         h = Graphlayout('adjMatrix',obj.adjMat); %'currentLayout',Treelayout());
       else
         h = Graphlayout('adjMatrix',obj.adjMat,'undirected',true); %'currentLayout',Treelayout());
       end
     
   end  
   
   function d = nnodes(obj)
     d = length(obj.adjMat);
   end
   
   function ns = neighbors(obj, i)
     ns = union(find(obj.adjMat(i,:)), find(obj.adjMat(:,i))');
   end
   
   function [d, pre, post, cycle, f, pred] = dfs(obj)
     % Depth first search - type 'help dfs' for details
     [d, pre, post, cycle, f, pred] = dfs(obj.adjMat, 1, obj.directed);
   end
   
    
  
   
   % We overload the syntax so that obj(i,j) refers to obj.adjMat(i,j)
   
   function B = subsref(obj, S)
     if (numel(S) > 1) % eg. obj.adjMat(1:3,:)
       B = builtin('subsref', obj, S);
     else
       switch S.type    %eg obj(1:3,:)
         case {'()'}
           B = obj.adjMat(S.subs{1}, S.subs{2});
         case '.' % eg. obj.adjMat
           B = builtin('subsref', obj, S);
       end
     end
   end
   
   function obj2 = subsasgn(obj, S, value)
     if (numel(S) > 1) % eg. obj.adjMat(1:3,:) = value
       obj2 = builtin('subsasgn', obj, S, value);
     else
       switch S.type    %eg obj(1:3,:)
         case {'()'}
           obj2 = obj;
           obj2.adjMat(S.subs{1}, S.subs{2}) = value;
         case '.' % eg. obj.adjMat = value
           obj2 = builtin('subsasgn', obj, S, value);
       end
     end
   end

 end
 
 
  
end