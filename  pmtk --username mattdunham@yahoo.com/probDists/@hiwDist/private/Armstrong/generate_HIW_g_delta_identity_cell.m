function [sigma_id, K_id]=generate_HIW_g_delta_identity_cell(g, cliques, delta)
% inputs: 1. g, the p x p symmetric matrix with
% respect to an original ordering v_1, ..., v_p
% 2. cliques, a 1 x t cell array of a perfect sequence of
% (nonempty) cliques of g,
% such as from chordal_to_ripcliques_cell.m
% output: 1. sigma_id, a random draw from HIW(g, delta, identity)
% 2. K_id=inv(sigma_id), a random draw from HW(g, delta, identity)
% THEORY: Roverato00 Theorem 3.
% This routine is first step in generating Sigma_i~HIW(g_i, delta, Phi_i).
index_finish=0; index_start=0;
num_cliques=size(cliques, 1);
[seps, residuals, histories]=seps_resids_hists(cliques);
% NOTE seps{1} will be [], because everyone writes them as S_2,...
num_Rj=zeros(1, num_cliques);
%% create perfect ordering as per C1, R2, R3,..definition
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
% now reverse the ordering, and use rev_perf=opposite
% order to perfect_order
% for indexing the graph and the parameter Phi
%% NOTE: can NOT assume that the ordering of g_reverse=
% my mcs ordering of g (had i computed it) in reverse. But
% by construction, it is opposite to a perfect order which
% satisfies at least one mcs with respect to g, and is constructed
% as C1, R2, R2, .., Rk as req’d by Roverato00p100
rev_perf=zeros(1,p);
index=0;
for i=1:p
  rev_perf(i)=perfect_order(p-index);
  index=index+1;
end
g_rev_perf=zeros(p);
g_rev_perf=g(rev_perf,rev_perf);
%%% to go back and forth: note that
% perf_order=p-rev_perf+1; perf+reverse=p+1
% Psi(i,i) is a random Chi_squared(delta+nu_i).
for i=1:p
  if i==p,
    Psi(p,p)=(chi2rnd(delta))^(.5);
  else
    % find nu_i=num of parents=adj. predecessors
    % of each vertex i wrt PERFECT
    % order, NOT reverse= an elimination order. So nu_p=0
    % because Psi is wrt reverse ordering.
    % nu_i, i>1 could be zero for disconnected case.
    clear nu_i
    nu_i=sum(g_rev_perf(i,i+1:n));
    Psi(i,i)=(chi2rnd(delta+nu_i))^(.5);
  end
end
% Psi(i,j), j >i is N(0,1) if an edge
% i,j exists in g, and zero otherwise.
for i=1:p-1
  for j=i+1:p
    if (g_rev_perf(i,j)==0)
      Psi(i,j)=0;
    else
      Psi(i,j)=randn;
    end
  end
end
K_rev_perf=zeros(p); sigma_id_rev_perf=zeros(p);
K_rev_perf=Psi’*Psi;
sigma_id_rev_perf=inv(K_rev_perf);
% is HIW(g_rev_perf, delta, id_rev_perf)
% Note that this is covariance of g wrt the rev_perf ordering,
% NOT wrt g. Should have zeros where off diagonal g_rev_perf does.
%%% Now need to do inverse permutation to get back to the Sigma for g
inverse_permute=zeros(1,p);
for j=1:p
  inverse_permute(j)=find(rev_perf==j);
end
K_id=zeros(p); sigma_id=zeros(p);
K_id=K_rev_perf(inverse_permute, inverse_permute);
sigma_id=sigma_id_rev_perf(inverse_permute, inverse_permute); % ~HIW(g, delta, id)
% Check that sigma_id has zeros in right place:
% inv_sigma_id=inv(sigma_id);
% indices=find(~(inv_sigma_id==0)); inv_sigma_id(indices)=1;
% note that indices is not pairs, but where (1,1)=1, (1,2)=2, etc.
%%%%
% NOTES
% ------
% it is critical that the adjacency matrix g_rev_perf
% is indexed OPPOSITE to the node ordering given by
% C_1, ,,,, C_k a PERFECT order of g (NOT g_elim),
% i.e. that given by C_1, R_2, R_3,... etc. (Roverato00
% p 100 2.1 para3 lines 3-> based on those preceding.)
% DESPITE the fact that any perfect mcs ordering
% is satisfies the reverse being a perfect mcs
% ordering also, regardless of the size of the
% last clique, or the first node you choose of the
% last clique to begin. And any perfect sequence
% of cliques can be reversed and it’s still a perfect
% sequence (so long as it was created under mcs)
% see my notes in folder "Generation: g must be wrt
% elim=perf1 opposite, but cliques with respect to
% perf1" and "Perfect=reverse Perfect" attached to "Perfect Orders".
% NOTE matlab orders union(a,b) as [min(a,b), max(a,b)]
% regardless of relative size of a,b
% so my cliques will always be ordered in strictly
% ascending order [min,..max]
% i.e. union([7,1,11,3], [4,9,18,2])
% becomes [1,2,3,4,7,9,11,18]
