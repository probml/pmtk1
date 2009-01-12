function [ln_h]=h_constant_ln_cell(cliques, delta, Phi)
% inputs: 1. cliques, a 1 x t cell array of the t nonempty
% cliques of g in RIP ordering
% (from chordal_to_ripcliques_cell.m)
% 2. delta, integer > 0
% 3. Phi>0, p x p parameter for Sigma_E~HIW(g, delta, Phi)
% output: 1. log of h(g, delta, Phi), the normalising constant for
% p(Sigma_E) where E is set of edges in g
% Appears in h_likelihood, the
% g-constrained likelihood p(Y=y |g).
% Y~ p-di Normal(0, inv(Omega))
% Based on Roverato 2000, Prop 2
ln_prod_top_terms=0;
ln_prod_bottom_terms=0;
seps=seps_resids_hists_cell(cliques);
for i=1:length(cliques)
C_i=cliques{i};
Phi_C_i=Phi(C_i, C_i);
numC_i=length(C_i);
ln_top_term_i= ( (delta+ numC_i -1) /2 ) * log(det(Phi_C_i/2) )...
- mvt_gamma_ln( numC_i, (delta+ numC_i -1) /2);
ln_prod_top_terms=ln_prod_top_terms+ln_top_term_i;
if i==1,
ln_bottom_term_i=0;
elseif isempty(seps{i});
% need this case for disconnected graph.
% If there are 2 components, then 2 empty seps
ln_bottom_term_i=0;
else
S_i=seps{i};
numS_i=length(S_i);
Phi_S_i=Phi(S_i, S_i);
ln_bottom_term_i=( (delta+ numS_i -1) /2 )* log(det(Phi_S_i/2))...
- mvt_gamma_ln( numS_i, (delta+ numS_i -1) /2);
end
ln_prod_bottom_terms=ln_prod_bottom_terms+ln_bottom_term_i;
end
ln_h=ln_prod_top_terms-ln_prod_bottom_terms;
% recall det(cA)=c^p det(A) where A is nxn
