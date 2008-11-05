
function ln_h = lognormconst(obj)
% Based on Roverato 2000, Prop 2
% Written by Helen Armstrong (see her thesis p63)
% Modified by Kevin Murphy
ln_prod_top_terms=0;
cliques = obj.G.cliques; seps = obj.G.seps;
delta = obj.delta; Phi = obj.Phi;
for i=1:length(cliques)
  C_i=cliques{i};
  Phi_C_i=Phi(C_i, C_i);
  numC_i=length(C_i);
  ln_top_term_i= ( (delta+ numC_i -1) /2 ) * log(det(Phi_C_i/2) )...
    - mvt_gamma_ln( numC_i, (delta+ numC_i -1) /2);
  ln_prod_top_terms=ln_prod_top_terms+ln_top_term_i;
end
ln_prod_bottom_terms=0;
for i=1:length(seps)
  S_i=seps{i};
  numS_i=length(S_i);
  Phi_S_i=Phi(S_i, S_i);
  ln_bottom_term_i=( (delta+ numS_i -1) /2 )* log(det(Phi_S_i/2))...
    - mvt_gamma_ln( numS_i, (delta+ numS_i -1) /2);
  ln_prod_bottom_terms=ln_prod_bottom_terms+ln_bottom_term_i;
end
ln_h=ln_prod_top_terms-ln_prod_bottom_terms;
%assert(isequal(ln_h, lognormconst2(obj)))
end

   