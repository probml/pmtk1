classdef VarElimInfEng < InfEng
% Sum-Product Variable Elimination 
    properties
        Tfac;
        domain;
        visVars;
        ordering;
        model;
    end
 
    methods
      
        function eng = condition(eng, model, visVars, visVals)    
            if(nargin < 4), visVars = []; visVals = {};end
            [eng.Tfac,nstates] = convertToTabularFactors(model,visVars,visVals);
            if isempty(eng.ordering)
                if(model.G.directed)
                    moralGraph = moralize(model.G);
                else
                    moralGraph = model.G;
                end
                eng.ordering = best_first_elim_order(moralGraph.adjMat,nstates);
            end
            eng.domain = model.domain;
            eng.visVars = visVars;
           
            model.infEng = [];
            eng.model = model;
            eng.model.infEng = eng;
        end
        
        function [postQuery,eng,Z] = marginal(eng, queryVars)
        % postQuery = sum_h p(Query,h)      
        
            if any(ismember(queryVars,eng.model.visVars))
               warning('VarElimInfEng:alreadyConditioned','You have already conditioned on one or more of your query variables.'); 
            end
            
            cfacs = (cellfun(@(x)isequal(pmf(x),1),eng.Tfac));                  % factors involving continuous, unobserved nodes
            if any(cfacs)
               cdom = cell2mat(cellfuncell(@(x)rowvec(sube(subd(x,'domain'),2)),eng.Tfac(cfacs))); % domain of unobserved continuous nodes
               I = intersect(cdom,queryVars);
               if ~isempty(I)
                   if numel(queryVars) > numel(I)
                       error('Cannot represent the joint over mixed node types.');
                   elseif numel(I) > 1
                       error('Cannot represent the joint over two or more unobserved continuous nodes.');
                   else
                        postQuery = extractLocalDistribution(eng.model,I);
                        assert( isa(postQuery,'MixtureDist'));
                        par = parents(eng.model.G, I);
                        assert(numel(par) == 1);
                        SS.counts = pmf(marginal(eng.model,par));
                        postQuery.mixingWeights = fit(postQuery.mixingWeights,'suffStat',SS);
                        return;
                   end
               end
            end
            elim = mysetdiff(mysetdiff(eng.domain(eng.ordering),queryVars),eng.visVars);
            [postQuery,Z] = normalizeFactor(VarElimInfEng.variableElimination(eng.Tfac,elim));
            
        end
        
        function [samples] = sample(eng,n)
           Tfac = marginal(eng,eng.domain);  
           % eng.Tfac may consist of unnormalized sliced factors after a call to
           % condition. Calling marginal(eng,eng.domain) here builds the full,
           % (conditioned) normalized table.
           samples = sample(Tfac,  n);
        end
        
        function logZ = lognormconst(eng)
             warning off VarElimInfEng:alreadyConditioned;
            [Tfac,eng,Z] = marginal(eng,eng.domain);
            logZ = log(Z);
            warning on VarElimInfEng:alreadyConditioned;
        end
        
    end
    
    
    methods(Static = true, Access = 'protected')
       
        function margFactor = variableElimination(factors,elimOrdering)
        % Perform sum-product variable elimination    
        % See Koller & Friedman algorithm 9.1 pg 273
            k = numel(elimOrdering);
            for i=1:k
               factors = eliminate(elimOrdering(i),factors); 
            end
            margFactor = TabularFactor.multiplyFactors(factors);
            
            
            function newFactors = eliminate(variable,factors)
            % eliminate a single variable    
                nfacs = numel(factors);
                inscope  = false(nfacs,1);   % inscope(f) is true iff the variable is in the scope of factors{f} 
                for f=1:nfacs
                    inscope(f) = any(variable==factors{f}.domain);
                end
                psi = TabularFactor.multiplyFactors(factors(inscope));
                tau = marginalize(psi,mysetdiff(psi.domain,variable)); % marginalize out the elimination variable
                newFactors = {factors{not(inscope)},tau};
            end
            
            
        end
        
        
    end
    
    
    
   
end