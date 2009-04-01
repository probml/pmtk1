function [treeAdjMat, wij] = chowliuTree(X, values)
% Find MLE undirected tree using Chow-Liu algorithm.
% X(i,j) is value of case i=1:n, node j=1:d
% values is set of valid values for each node (e.g., [0 1])
% treeAdjMat is a sparse matrix
% wij is a d*d mutual information matrix
%
% O(N d^2) time to compute p(i,j), N=#cases, d=#nodes.
% O(d^2 K^2) tome to compute MI, K=#states
% O(d^2) time to find MWST

%#author Sam Roweis
%#modified Kevin Murphy

% This implementation is efficient since it only ever uses
% for loops over the states K, which are often binary.
% There is no loop over n or d.

% For fast kernel density estimation methods for MI
% see "Fast Calculation of Pairwise Mutual Information for
%Gene Regulatory Network Reconstruction"
%http://www.stanford.edu/~qiupeng/pdf/FastPairMI_revision.pdf

documents = X';
[numvar numdoc] = size(documents); %variables/words by documents
numval = length(values);


% collect counts and calculate joint probabilities
% pij(x1,x2,v1,v2) = prob(x1=values(v1),x2=values(v2))
pij = zeros(numvar,numvar,numval,numval);
for v1=1:numval,
  for v2=1:numval,
    %pij(:,:,v1,v2) = full((documents==values(v1))*(documents==values(v2))');
    A = double(documents==values(v1));
    B = double(documents==values(v2));
    % A(x1,d) = 1 iff D(x1,d)=v1,  B(x2,d) = 1 iff D(x2,d) = v2
    % pij(x1,x2,v1,v2) = sum_d A(x1,d)  B(x2,d) = A*B'
    pij(:,:,v1,v2) = A*B';
  end;
end;
pij = pij/numdoc;

% calculate marginal probabilities
% pi(x1,v) = pij(x1,x1, v,v)
pi2 = reshape(pij,numvar^2,numval^2);
pi = pi2(find(eye(numvar)),find(eye(numval)));

% Calculate entropies and mutual information
% We need to avoid log of 0.
% if pi(x,v)=0 for all v, then entropy(pi(x,:)) = 0
% since -sum_v pi(x,v) log pi(x,v) = 0
% Hence it is safe to replace 0 with eps inside the log
minprob = 1/numdoc; % eps;
hi  = -sum(pi.*log(max(pi,minprob)),2);
hi  = hi(:,ones(1,numvar));
hij = -sum(sum(pij.*log(max(pij,minprob)),3),4);
wij = -hij+hi+hi'; % mutual information

treeAdjMat = minimumSpanningTree(-wij); % find max weight spanning tree
treeAdjMat = setdiag(treeAdjMat, 0);


