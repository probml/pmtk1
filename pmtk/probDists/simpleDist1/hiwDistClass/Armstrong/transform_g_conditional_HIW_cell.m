function [sigma_D, K_D]=transform_g_conditional_HIW_cell(sigma_B, g, cliques, delta, B, D)
% inputs: 1. sigma_B~HIW(g, delta, B), the p x p covariance
% 2. g, the p x p symmetric matrix with
% respect to an original ordering v_1, ..., v_p
% 3. cliques, a 1 x t cell array of a perfect sequence of
% (nonempty) cliques of g,
% such as from chordal_to_ripcliques_cell.m
% 4. delta, the degrees of freedom parameter of the distribution of sigma_B
% 5. B, the matrix parameter of the distribution of sigma_B
% 6. D, the matrix parameter of the distribution of the transformed covariance
% output: 1. sigma_D, a random draw from HIW(g, delta, D)
% 2. K_D=inv(sigma_D), a random draw from HW(g, delta, D)
% THEORY: Roverato00 Theorem 4.
% sigma_B~g-conditional HIW(g, delta, B)
% to sigma_D~g-conditional HIW(g, delta, D)
% K_D is inv sigma_D
% NOTE B and D constrained to satisfy inv(B)(i,j)=0 iff g(i,j)=0.
% use to generate a random draw=Sigma_i~HIW(g_i, delta, Phi_i),
% where Phi_i are iterate
% outputs of the mcmc for each graph iterate g_i.
p=length(g);
num_cliques=length(cliques);
[seps, residuals, histories]=seps_resids_hists(cliques);
% NOTE the seps{1} will be [], because everyone writes them as S_2,...
num_Cj=zeros(1, num_cliques);
num_Sj=zeros(1, num_cliques); %num_Sj(1,1)=num_Rj(1,1)=0;
num_Rj=zeros(1, num_cliques);
num_Cj(1,1)=length(cliques{1});
%%% create perfect ordering as per C1, R2, R3,..definition
perfect_order=zeros(1,p);
perfect_order(1:length(cliques{1}))=cliques{1};
index_finish=length(cliques{1});
for j=2:num_cliques
    Cj=cliques{j}; Rj=residuals{j}; Sj=seps{j};
    num_Cj(1,j)=length(Cj);num_Rj(1,j)=length(Rj);num_Sj(1,j)=length(Sj);
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
g_rev_perf=g(rev_perf,rev_perf);
B_rev_perf=B(rev_perf,rev_perf);
D_rev_perf=D(rev_perf,rev_perf);
%%%% to go back and forth: note that
% perf_order=p-rev_perf+1; perf+reverse=p+1
%%%% transformation as per thm 4 roverato2000
% sigma_B~HIW(g_rev_perf, delta, B_rev_perf) to
% Sigma_D~HIW(g_rev_perf, delta, D_rev_perf)
% Let Matrix_cj be the sub matrix of Matrix indexed by cliques of
% g NOT g_elim, and similarly for the separators and residuals.
sigma_B_rev_perf=sigma_B(rev_perf,rev_perf);
K_B=inv(sigma_B_rev_perf);
choleskyK_B=chol(K_B);
c1=num_Cj(1,1);
clear index_start index_finish
index_start=p-c1+1; index_finish=p;
indexC1_in_Upsilon_D=zeros(1,c1);
indexC1_in_Upsilon_D=[index_start: index_finish];
% this is the last c1 columns
Upsilon_D=zeros(p);
B_1=zeros(c1); D_1=zeros(c1);
Q_1=zeros(c1); P_1=zeros(c1); O_1=zeros(c1);
B_1=B_rev_perf(indexC1_in_Upsilon_D,indexC1_in_Upsilon_D);
Q_1=chol(inv(B_1));
D_1=D_rev_perf(indexC1_in_Upsilon_D,indexC1_in_Upsilon_D);
P_1=chol(inv(D_1));
O_1=inv(Q_1)*P_1; % this is a letter O
Upsilon_D(indexC1_in_Upsilon_D, indexC1_in_Upsilon_D)=...
    choleskyK_B(indexC1_in_Upsilon_D,indexC1_in_Upsilon_D)*O_1;
for j=2:num_cliques
    clear cj indexCj_in_Upsilon_D
    cj=num_Cj(j);
    indexCj_in_Upsilon_D=zeros(1, cj);
    clear Rj rj indexRj_inOj indexRj_in_Upsilon_D
    clear unsort_indexRj_in_Upsilon_D
    Rj=residuals{j};
    rj=num_Rj(j);
    indexRj_inOj=zeros(1,rj);
    indexRj_in_Upsilon_D=zeros(1,rj);
    unsort_indexRj_in_Upsilon_D=zeros(1,rj);
    clear Sj sj indexSj_inOj indexSj_in_Upsilon_D
    clear unsort_indexSj_in_Upsilon_D
    Sj=seps{j};
    sj=num_Sj(j);
    indexSj_inOj=zeros(1,sj);
    indexSj_in_Upsilon_D=zeros(1,sj);
    unsort_indexSj_in_Upsilon_D=zeros(1,sj);
    B_j=zeros(cj); D_j=zeros(cj);
    Q_j=zeros(cj); P_j=zeros(cj); O_j=zeros(cj); % this is a letter O
    for k=1:rj
        unsort_indexRj_in_Upsilon_D(k)=find(rev_perf==Rj(k));
    end
    indexRj_in_Upsilon_D=sort(unsort_indexRj_in_Upsilon_D);
    for k=1:sj
        unsort_indexSj_in_Upsilon_D(k)=find(rev_perf==Sj(k));
    end
    indexSj_in_Upsilon_D=sort(unsort_indexSj_in_Upsilon_D);
    indexCj_in_Upsilon_D=[indexRj_in_Upsilon_D, indexSj_in_Upsilon_D];
    B_j=B_rev_perf(indexCj_in_Upsilon_D,indexCj_in_Upsilon_D);
    Q_j=chol(inv(B_j));
    D_j=D_rev_perf(indexCj_in_Upsilon_D,indexCj_in_Upsilon_D);
    P_j=chol(inv(D_j));
    O_j=inv(Q_j)*P_j;
    indexRj_inOj=1:rj;
    indexSj_inOj=rj+1:cj;
    Upsilon_D(indexRj_in_Upsilon_D, indexRj_in_Upsilon_D)=...
        choleskyK_B(indexRj_in_Upsilon_D, indexRj_in_Upsilon_D)*...
        O_j(indexRj_inOj, indexRj_inOj);
    Upsilon_D(indexRj_in_Upsilon_D,indexSj_in_Upsilon_D)=...
        choleskyK_B(indexRj_in_Upsilon_D,indexRj_in_Upsilon_D)*...
        O_j(indexRj_inOj,indexSj_inOj)+...
        choleskyK_B(indexRj_in_Upsilon_D,indexSj_in_Upsilon_D)*...
        O_j(indexSj_inOj,indexSj_inOj);
end
clear Rj rj indexRj_inOj indexRj_in_Upsilon_D
clear unsort_indexRj_in_Upsilon_D
Rj=residuals{j};
rj=num_Rj(j);
indexRj_inOj=zeros(1,rj);
indexRj_in_Upsilon_D=zeros(1,rj);
unsort_indexRj_in_Upsilon_D=zeros(1,rj);
clear Sj sj indexSj_inOj indexSj_in_Upsilon_D
clear unsort_indexSj_in_Upsilon_D
Sj=seps{j};
sj=num_Sj(j);
indexSj_inOj=zeros(1,sj);
indexSj_in_Upsilon_D=zeros(1,sj);
unsort_indexSj_in_Upsilon_D=zeros(1,sj);
B_j=zeros(cj); D_j=zeros(cj);
Q_j=zeros(cj); P_j=zeros(cj); O_j=zeros(cj); % this is a letter O
for k=1:rj
    unsort_indexRj_in_Upsilon_D(k)=find(rev_perf==Rj(k));
end
indexRj_in_Upsilon_D=sort(unsort_indexRj_in_Upsilon_D);
for k=1:sj
    unsort_indexSj_in_Upsilon_D(k)=find(rev_perf==Sj(k));
end
indexSj_in_Upsilon_D=sort(unsort_indexSj_in_Upsilon_D);
indexCj_in_Upsilon_D=[indexRj_in_Upsilon_D, indexSj_in_Upsilon_D];
B_j=B_rev_perf(indexCj_in_Upsilon_D,indexCj_in_Upsilon_D);
Q_j=chol(inv(B_j));
D_j=D_rev_perf(indexCj_in_Upsilon_D,indexCj_in_Upsilon_D);
P_j=chol(inv(D_j));
O_j=inv(Q_j)*P_j;
indexRj_inOj=1:rj;
indexSj_inOj=rj+1:cj;
Upsilon_D(indexRj_in_Upsilon_D, indexRj_in_Upsilon_D)=...
    choleskyK_B(indexRj_in_Upsilon_D, indexRj_in_Upsilon_D)*...
    O_j(indexRj_inOj, indexRj_inOj);
Upsilon_D(indexRj_in_Upsilon_D,indexSj_in_Upsilon_D)=...
    choleskyK_B(indexRj_in_Upsilon_D,indexRj_in_Upsilon_D)*...
    O_j(indexRj_inOj,indexSj_inOj)+...
    choleskyK_B(indexRj_in_Upsilon_D,indexSj_in_Upsilon_D)*...
    O_j(indexSj_inOj,indexSj_inOj);
end
K_D_rev_perf=zeros(p); sigma_D_rev_perf=zeros(p);
K_D_rev_perf=Upsilon_D'*Upsilon_D; % ~HW(g_rev_perf, delta, D_rev_perf)
sigma_D_rev_perf=inv(K_D_rev_perf); % ~HIW(g_rev_perf, delta, D_rev_perf)
% Note that this is covariance of g wrt the rev_perf ordering, NOT wrt g.
%%% Now need to do inverse permutation to get back to the Sigma for g
inverse_permute=zeros(1,p);
for j=1:p
    inverse_permute(j)=find(rev_perf==j);
end
K_D=zeros(p);
sigma_D=zeros(p);
K_D=K_D_rev_perf(inverse_permute, inverse_permute); % ~Wishart(g, delta+p-1, D) (E(K_D) prop inv(D))
sigma_D=sigma_D_rev_perf(inverse_permute, inverse_permute); % ~HIW(g, delta, D) (E(D) prop D