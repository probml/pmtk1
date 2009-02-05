%% Example of sampling from the HIW distribution
%#testPMTK
% Consider graph on Armstrong p27
A = zeros(16,16);
ndx = [1 2 6 11]; A(ndx, ndx) = 1;
ndx = [4 7 10 14 15]; A(ndx,ndx) = 1;
ndx = [3 5 8 9 12 13 16]; A(ndx, ndx) = 1;
A = setdiag(A,0);
G = ChordalGraph(A);
d = size(A,1);
n = 100;
Sigma = randpd(d);
X = mvnrnd(zeros(1,d), Sigma, n);
Sy = cov(X);
Phi = 0.1*Sy;
delta = 5;
obj = HiwDist(G, delta, Phi);
L = lognormconst(obj);
S = sample(obj, 1);
K = inv(S);
% we should have K(i,j)=0 iff G(i,j)=0
figure;
subplot(2,2,1); imagesc(G.adjMat);
subplot(2,2,2); imagesc(K)
% Also, K should look like fig 2.1 on p24, if in perfect order
po = G.perfectElimOrder;
subplot(2,2,3); imagesc(G.adjMat(po,po));
subplot(2,2,4); imagesc(K(po,po))