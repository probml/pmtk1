function margFactor = variableElimination(factors,elimOrdering)
% Perform sum-product variable elimination
% See Koller & Friedman algorithm 9.1 pg 273
%
% Returns a single TabularFactor representing the *unnormalized* joint
% distribution over the variables in the scope of the input factors but not in
% the elimOrdering vector. To normalize the factor call 
% normalizeFactor(margFactor).


    k = numel(elimOrdering);
    for i=1:k
        factors = eliminate(elimOrdering(i),factors);
    end
    margFactor = TabularFactor.multiplyFactors(factors);
    
    
    function newFactors = eliminate(variable,factors)
        % eliminate a single variable
        inscope = cellfun(@(fac)any(variable == fac.domain),factors);           % inscope(f) is true iff the variable is in the scope of factors{f}
        psi = TabularFactor.multiplyFactors(factors(inscope));
        tau = marginalize(psi,mysetdiff(psi.domain,variable));                  % marginalize out the elimination variable
        newFactors = {factors{not(inscope)},tau};
    end
    
    
end