classdef graph 
  
  properties
    adjMat;
  end
  
 %methods(Abstract = true)
 %  e = nedges(obj);
 %  ns = neighbors(obj, v);
 %end

 %% Main methods
 methods
   function obj = graph(adjMat)
     if nargin == 0
       obj.adjMat = [];
     else
       obj.adjMat = adjMat;
     end
   end
   
   function h=draw(obj)
     h = graphlayout('adjMatrix',obj.adjMat);
   end
   
   %{
   function h=draw(obj)
     % Use graphviz to layout graph and visualize it
     % If you have the bioinformatics toolbox, you can edit the
     % resulting layout eg.
     % h = draw(graph(rand(5,5)>0.5));
     % set(h,'layouttype','hierarchical')
     % dolayout(h)
     if bioToolboxInstalled
       d = length(obj.adjMat);
       for i=1:d, names{i}=sprintf('%d', i); end
       biog = biograph(obj.adjMat, names);
       h=view(biog);
       set(h,'layouttype', 'equilibrium')
       dolayout(h)
     else
       drawGraph(obj.adjMat);
     end
   end
   %}
   
   function d = nnodes(obj)
     d = length(obj.adjMat);
   end
   
   function ns = neighbors(obj, i)
     ns = union(find(obj.adjMat(i,:)), find(obj.adjMat(:,i))');
   end
   
   function [d, pre, post, cycle, f, pred] = dfs(obj, start, directed)
     % Depth first search
     % Input:
     % adj_mat(i,j)=1 iff i is connected to j.
     % start is the root vertex of the dfs tree; if [], all nodes are searched
     % directed = 1 if the graph is directed
     %
     % Output:
     % d(i) is the time at which node i is first discovered.
     % pre is a list of the nodes in the order in which they are first encountered (opened).
     % post is a list of the nodes in the order in which they are last encountered (closed).
     % 'cycle' is true iff a (directed) cycle is found.
     % f(i) is the time at which node i is finished.
     % pred(i) is the predecessor of i in the dfs tree.
     %
     % If the graph is a tree, preorder is parents before children,
     % and postorder is children before parents.
     % For a DAG, topological order = reverse(postorder).
     %
     % See Cormen, Leiserson and Rivest, "An intro. to algorithms" 1994, p478.
     [d, pre, post, cycle, f, pred] = dfsHelper(obj.adjMat, start, directed);
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