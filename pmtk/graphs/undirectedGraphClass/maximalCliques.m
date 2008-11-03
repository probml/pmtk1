function complete_subgraphs = maximalCliques( X )
%MAXIMALCLIQUES finds all the maximal complete sub-graphs in a graph
%   The graph passed must be an upper rectangular square matrix. Each row
%   of a matrix denotes the presence of an edge with 1, and an absence of
%   it with 0. The row and col no. of an edge denotes the connecting nodes.
%   Given this matrix, this function finds all the maximal complete 
%   sub-graph (a set of nodes amongst all the nodes which form a complete 
%   sub-graph i.e. every node connects to every other) also known as 
%   cliques. A maximal graph is returned since every complete sub-graph 
%   will also have smaller complete sub-graphs inside itself. NOTE: this 
%   function would not return single node sub-graphs, although every
%   isolated node, in concept, also forms a complete sub-graph
%   The function returns all the sub-graphs in a cell-array, where each
%   row denotes a new sub-graph

%#author Ahmad
%#url lhttp://www.mathworks.com/matlabcentral/fx_files/19889/1/maximalCliques.m

%TEST CASES

% A = [-1  1  1  0  0  0
%      -1 -1  1  0  0  0
%      -1 -1 -1  1  0  1 
%      -1 -1 -1 -1  0  1
%      -1 -1 -1 -1 -1  0
%      -1 -1 -1 -1 -1 -1 ];

% B = [-1  1  1  0  1  1
%      -1 -1  1  1  1  1
%      -1 -1 -1  1  1  1 
%      -1 -1 -1 -1  1  0
%      -1 -1 -1 -1 -1  1
%      -1 -1 -1 -1 -1 -1 ];
%
% C = [-1  0  0  0  0  0  0  0  0
%      -1 -1  1  1  0  0  0  0  0
%      -1 -1 -1  0  0  0  0  0  0
%      -1 -1 -1 -1  0  0  0  0  0
%      -1 -1 -1 -1 -1  0  0  0  0
%      -1 -1 -1 -1 -1 -1  0  1  0
%      -1 -1 -1 -1 -1 -1 -1  0  1
%      -1 -1 -1 -1 -1 -1 -1 -1  1
%      -1 -1 -1 -1 -1 -1 -1 -1 -1 ];

    [m n] = size(X);
    
    assert( m == n, 'The matrix should be square, cause each side denotes the nodes in the same graph' );

    %the graph 
    complete_subgraphs = {};
    
    %this will keep track of all the nodes that need to be looked at
    remaining_nodes = 1:n;
    
    %discard the lower triangular matrix, just in case it has values
    X = X - tril(X);
    
    connecting_nodes_set = {};
    
    %sort nodes by the decreasing number of edges
    for i = 1:n
        connecting_nodes = returnConnectingNodes(X, i);
        connecting_nodes_set = [connecting_nodes_set; {connecting_nodes}];
    end
    [idx, remaining_nodes] = sort( cellfun(@length, connecting_nodes_set), 'descend' );
    
    %main loop
    while ~isempty(remaining_nodes)
        
        %choose just the first node, in the possible nodes
        source_node = remaining_nodes(1);
        
        %remove from remaining_nodes
        %remaining_nodes = setdiff(remaining_nodes, source_node);
        remaining_nodes = remaining_nodes(2:end);
        
        %find all nodes with whom the source node has an edge
        %connecting_nodes = returnConnectingNodes( X, source_node );
        connecting_nodes = connecting_nodes_set{source_node};
        
        group_size = length(connecting_nodes);
        
        %add all maximal subgraphs related to this node, which haven't been
        %already added
        
        %check only if there are some connecting nodes remaining, and the
        %group_size is more than 0
        while  ~isempty(connecting_nodes) && group_size > 0
            %find out all the valid connecting_node combinations at a certain size
            cmbs = nchoosek(connecting_nodes, group_size);
            
            %loop through all combinations computed
            for i = 1:size(cmbs, 1)
                cmb = cmbs(i,:);
                
                %check if a given combination makes a maximal subgraph or
                %has only a single node (i.e. two nodes including source)
                if group_size == 1 || checkIfMaximalSubgraph( connecting_nodes_set, cmb )
                    %if it does add to complete subgraphs after attaching
                    %source_node
                    new_complete_subgraph = [source_node cmb];
                    complete_subgraphs = [ complete_subgraphs; sort(new_complete_subgraph) ];

                    %also remove all points from the set of connecting_nodes
                    connecting_nodes = setdiff(connecting_nodes, cmb);
                    
                    %also remove all points from the set of connecting_nodes
                    %remaining_nodes = setdiff(remaining_nodes, cmb);
                    for j = 1:length(cmb)
                        remaining_nodes(remaining_nodes == cmb(j)) = [];
                    end
                end
            end
            
            %decrease the group size
            group_size = group_size - 1;
        end
        
    end
    
end


function can_be_added = checkIfMaximalSubgraph( connecting_nodes_set, subgraph )
%Returns 1 if the every node in the subgraph has edges to all the other 
%nodes. This would make it a complete subgraph

    %initialize the return variable
    can_be_added = 0;
    
    %loop till we have one node remaining
    while length(subgraph) > 1
        %choose the first node left over
        node = subgraph(1);
        
        %delete it from the graph
        subgraph(subgraph == node) = [];
        
        %check if its connected to the rest of the graph
        if checkNodeForMaximalSubgraph(connecting_nodes_set, subgraph, node) == 0
            %if not, return a zero
            return;
        end
    end
    
    %return success. All nodes connected to each other
    can_be_added = 1;
end

function can_be_added = checkNodeForMaximalSubgraph( connecting_nodes_set, complete_subgraph, test_node )
%Returns if the given test_node has edges to all the nodes in the
%complete_subgraph

    %find the connecting nodes of the test node
    %test_node_edges = returnConnectingNodes( X, test_node );
    test_node_edges = connecting_nodes_set{test_node};
    
    %see if all the nodes already present in the subgraph, are also present
    %in the connecting nodes of the test node
    can_be_added = all( ismember( complete_subgraph, test_node_edges ) );
end


function connecting_nodes = returnConnectingNodes( X, node_no )
%Returns the nodes to which the node_no have edges

    row_psbs = find( X(node_no,:) );        %search row for psbs
    col_psbs = find( X(:,node_no) )';       %search col for psbs
    
    %combine all the edges and return
    connecting_nodes = union( row_psbs, col_psbs );
end
