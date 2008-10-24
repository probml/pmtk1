function testHelen()

% Is the perfect order code on p221 of Armstrong necessary?
% No!

% Do examples from Helen Armstrong's thesis p27
A = zeros(16,16);
ndx = [1 2 6 11]; A(ndx, ndx) = 1;
ndx = [4 7 10 14 15]; A(ndx,ndx) = 1;
ndx = [3 5 8 9 12 13 16]; A(ndx, ndx) = 1;
A = setdiag(A,0);


%{
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
%}

G = chordalGraph(A)
assert(G.ischordal)
G.perfectElimOrder

cliques = G.cliques;
g = G.adjMat;
index_finish=0; index_start=0;
num_cliques=length(cliques); % size(cliques, 1);
[seps, residuals, histories]=seps_resids_hists_cell(cliques);
% NOTE seps{1} will be [], because everyone writes them as S_2,...
num_Rj=zeros(1, num_cliques);
% create perfect ordering as per C1, R2, R3,..definition
p=length(g);
perfect_order=zeros(1,p);
perfect_order(1:length(cliques{1}))=cliques{1};
index_finish=length(cliques{1});
for j=2:num_cliques
  Rj=residuals{j};
  num_Rj(1,j)=length(Rj);
  index_start=index_finish+1;
  index_finish=index_start+num_Rj(j)-1;
  perfect_order(index_start:index_finish)=Rj;
end
assert(isequal(G.perfectElimOrder, perfect_order))


rev_perf=zeros(1,p);
index=0;
for i=1:p
  rev_perf(i)=perfect_order(p-index);
  index=index+1;
end
isequal(rev_perf, G.perfectElimOrder(end:-1:1))
