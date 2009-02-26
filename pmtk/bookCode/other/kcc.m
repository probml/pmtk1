% [idx,dpsim]=kcc(S,k,nruns,maxits)
%
% Performs k-centers clustering (aka k-medoids clustering) to find k
% data centers (data points), by starting with a randomly selected
% set of k centers and iteratively refining this set using up to a
% maximum of maxits iterations. For N data points, the input is an NxN
% similarity matrix, s. s(i,k) is the similarity that data point i has
% to data point k (its potential cluster center). Alternatively, s may
% be an Mx3 matrix where each row contains two data point indices and
% a similarity value. M is the number of measured pair-wise input
% similarities and s(j,3) is the similarity of data point s(j,1) to
% point s(j,2). Similarities are assumed to be additivie (so,
% if they are computed from a probability model, use log-probabilities.)
% The iterative procedure terminates if the updated set of exemplars is
% identical to the set of exemplars identified in the previous iteration.
% 
% Input:
%
%   S         Similarity matrix (see above)
%   K         Number of clusters to find, or a vector of indices
%             of the initial set of exemplars
%   nruns     Number of runs to try, where each run is initialized
%             randomly (default 1)
%   maxits    Maximum number of iterations (default 100)
%
% Ouput:
%
%   idx(i,j)    Index of the data point that data point i is assigned
%               to in the jth run. idx(i,j)=i indicates that point i
%               is an exemplar
%   dpsim(m,j)  Sum of similarities of data points to exemplars, after
%               iteration m in the jth run
%
% Copyright Brendan J. Frey and Delbert Dueck, Aug 2006. This software
% may be freely used and distributed for non-commercial purposes.
%

function [idx,dpsim]=kcc(S,K,nruns,maxits);

if nargin<2 error('kcc:1','Too few input arguments');
elseif nargin==2 nruns=1; maxits=100;
elseif nargin==3 maxits=100;
elseif nargin>4 error('kcc:2','Too many input arguments');
end;
if length(K)==1 k=K; ui=0; else k=length(K); ui=1; end;

n=size(S,1); dpsim=zeros(maxits,nruns); idx=zeros(n,nruns);
for rep=1:nruns
    if ui mu=K; else tmp=randperm(n)'; mu=tmp(1:k); end;
    i=0; dn=(i==maxits);
    while ~dn
        i=i+1; muold=mu; dpsim(i,rep)=0;
        [tmp cl]=max(S(:,mu),[],2); % Find class assignments
        cl(mu)=1:k; % Set assignments of exemplars to themselves
        for j=1:k % For each class, find new exemplar
            I=find(cl==j);
            [Scl ii]=max(sum(S(I,I),1));
            dpsim(i,rep)=dpsim(i,rep)+Scl(1);
            mu(j)=I(ii(1));
        end;
        if (sum(sort(muold)==sort(mu))==k)||(i==maxits) dn=1; end;
    end;
    idx(:,rep)=mu(cl); dpsim(i+1:end,rep)=dpsim(i,rep);
end;
