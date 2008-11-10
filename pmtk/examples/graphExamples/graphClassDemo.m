%% Graph Demo
% Do the example in fig 23.4 p479 of Cormen, Leiserson and Rivest (1994)

u = 1; v = 2; w = 3; x = 4; y = 5; z = 6;
n = 6;
G=zeros(n,n);
G(u,[v x])=1;
G(v,y)=1;
G(w,[y z])=1;
G(x,v)=1;
G(y,x)=1;
G(z,z)=1;

% u1 -> v2  w3
% |    ^ |  / |
% |  /   | /  |
% v      v    v
% x4<-- y5   z6 (self)

GG = DirectedGraph(G);
draw(GG)
[d, pre, post, cycle, f, pred] = dfs(GG,[],1);
assert(isequal(d, [1 2 9 4 3 10]))
assert(isequal(f, [8 7 12 5 6 11]))
assert(cycle)

% break self loop
%GG(z,z)=0;
GG.adjMat(z,z) =0;
assert(~checkAcyclic(GG))

% break y->x edge, leaving only undirected cycle uvx

% u1 -> v2  w3
% |    ^ |  / |
% |  /   | /  |
% v      v    v
% x4    y5   z6

%GG(y,x) = 0;
GG.adjMat(y,x)=0;
assert(checkAcyclic(GG))
G1 = Dag(GG.adjMat);
G1.topoOrder % [3 6 1 4 2 5]


% Now give it an undirected cyclic graph
G = UndirectedGraph('type', 'lattice2D', 'nrows', 2, 'ncols', 2, 'wrapAround', 0);
% 1 - 3
% |   |
% 2 - 4
assert(~checkAcyclic(G))

% Now break the cycle
%G(1,2)=0; G(2,1)=0;
G.adjMat(1,2)=0; G.adjMat(2,1)=0;
assert(checkAcyclic(G))

% Make all UGs
%UGs = mkAllUG(UndirectedGraph(), 5);


% Test min span tree using example from Cormen, Leiserson, Rivest 1997 p509
A = zeros(9,9);
A(1,2)=4; A(2,3) = 8; A(3,4) = 7; A(4,5) = 9; A(4,6) = 14; A(5,6)=10;
A(1,8)=8; A(8,9)=7; A(9,3)=2; A(9,7)=6; A(8,7)=1; A(3,6)=4; A(7,6)=2;
A = mkGraphSymmetric(A);
[T,cost] = minSpanTree(UndirectedGraph(A))
assert(cost==37)