function l = isleaf(adj_mat, i)
% Return true if the node is a leaf (has no children)

l = isempty(children(adj_mat,i));
