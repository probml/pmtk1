function d = descendants(adj_mat,i)
% Recursively find all descendants of a a node, (all children, grandchildren, ... etc)

    d = [];
    if isleaf(adj_mat,i)
      return;
    else
       c = children(adj_mat,i);
       for i=1:numel(c)
           d = [d,descendants(adj_mat,c(i))];
       end
       d = [c,d];
    end
    
    
    
    
end