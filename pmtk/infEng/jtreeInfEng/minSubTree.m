function subtree = minSubTree(tree, nodes)
% Return the minimum subtree of tree that contains the specified nodes
% tree and subtree are adjacency matrices
% nodes is an array of indices indicating which nodes in tree must must be in the
% subtree.

    tree = triu(mkSymmetric(tree));
    root = sub(1:size(tree,1),not(sum(tree,1)));
    if(numel(root) ~= 1)
        error('Not a tree!'); % note - not a sufficient condition
    end
    
    if isempty(tree) || isempty(nodes)
        subtree = [];
        return;
    end
    
    rnodes = min_subtree_nodes(tree, nodes);
    node_num = length(tree);
    subtree = zeros(node_num);
    subtree(rnodes, rnodes) = tree(rnodes, rnodes);
   
    
    
function rnodes = min_subtree_nodes(tree, nodes)
    rnodes = [];
    if isempty(tree) || isempty(nodes)
        return
    end
    
    rnodes = nodes(1);
    newnodes = neighbors(tree, nodes(1));
    while ~issubset(nodes, rnodes)
        swapnodes = newnodes;
        newnodes = [];
        added = 0;
        for i=1:length(swapnodes)
            inode = swapnodes(i);
            tnodes = union(inode, rnodes);
            if issubset(nodes, tnodes)
                added = 1;
                break;
            end
            nns = neighbors(tree, inode);
            add_nodes = mysetdiff(nns, tnodes);
            newnodes = union(newnodes, add_nodes);
        end
        if added
            rnodes = tnodes;
        else
            rnodes = union(rnodes, newnodes);
        end
    end
    
function nea_node = nearest_node(tree, inode, nodes)
    nea_node = [];
    if ismember(inode, nodes)
        nea_node = inode;
        return;
    end
    cs = children(tree, inode);
    for i = 1:length(cs)
        n = cs(i);
        nea_node = nearest_node(tree, n, nodes);
    end
    
    
    
