%% Make a Chordal Graph
%  --  1
% |   /  \
% 4- 2 - 3
%    \  /
%      5
%      | 
%      6
%#testPMTK
A= zeros(6,6);
A(1,[2 3 4])=1;
A(2,[1 3 4 5])=1;
A(3,[1 2 5])=1;
A(4,[1 2])=1;
A(5,[2 3 6])=1;
A(6,5)=1;
G = ChordalGraph(A);
%G.cliques
assert(G.ischordal)

% Now make it non-chordal
%%G(2,3)=0; % also sets G(3,2)=0 for free %%%% BUG
%G2 = ChordalGraph(G.adjMat);
A(2,3)=0; A(3,2) = 0;
G2 = ChordalGraph(A);
assert(~G2.ischordal)

% Now make it chordal
G3 = ChordalGraph(G.adjMat, 'makeChordal');
assert(G3.ischordal)
G3.fillInEdges;

% Now make Jtree
J = Jtree(G.adjMat);


% Do examples from Helen Armstrong's thesis p27
A = zeros(16,16);
ndx = [1 2 6 11]; A(ndx, ndx) = 1;
ndx = [4 7 10 14 15]; A(ndx,ndx) = 1;
ndx = [3 5 8 9 12 13 16]; A(ndx, ndx) = 1;
A = setdiag(A,0);
figure;imagesc(1-A);colormap('gray')
G = ChordalGraph(A);
assert(G.ischordal)

perm = G.perfectElimOrder;
B = A(perm,perm);
figure;imagesc(1-B);colormap('gray')

% p29
A = zeros(26,26);
ndx = [8 17 22 14 16 25 15];
A(ndx, 20)=1;
for i=1:(length(ndx)-1)
  A(ndx(i),ndx(i+1))=1;
end
ndx = [10 6 18]; A(ndx,ndx)=1;
ndx = [6 7 18]; A(ndx,ndx)=1;
ndx = [6 18 2]; A(ndx,ndx) = 1;
ndx = [11 3 13 23]; A(ndx,ndx) = 1;
ndx = [23 13 24 12 26]; A(ndx,ndx)=1;
ndx = [12 24 5]; A(ndx,ndx)=1;
ndx = [26 12 5]; A(ndx,ndx)=1;
A(24,1)=1;
A(5,19)=1;
ndx = [21 23 26]; A(ndx,ndx)=1;
ndx = [9 4 21]; A(ndx,ndx)=1;
A = mkGraphSymmetric(A);
A = setdiag(A,0);
figure;imagesc(1-A);colormap('gray')
G = ChordalGraph(A);
assert(G.ischordal)

perm = G.perfectElimOrder;
B = A(perm,perm);
figure;imagesc(1-B);colormap('gray')

% Now make it non-chordal (p30)
A(8,15) = 1; A(15,8)=1;
figure;imagesc(1-A);colormap('gray')
G = ChordalGraph(A);
assert(~G.ischordal)
