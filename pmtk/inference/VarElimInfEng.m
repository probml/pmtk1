classdef VarElimInfEng < InfEng
% Sum-Product Variable Elimination 
    properties
        Tfac;
        domain;
        visVars;
        ordering;
        model;
        verbose;
    end
 
    methods
      
      function eng = VarElimInfEng(varargin)
        [eng.verbose] = process_options(varargin, ...
          'verbose', false);
      end
      
        function [eng, logZ, other] = condition(eng, model, visVars, visVals)    
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
            eng.model = model;
            if nargout >= 2
              [postQuery,eng,Z] = marginal(eng, []);
              logZ = log(Z);
            end
            other = [];
        end
        
     
        function [postQuery,eng,Z] = marginal(eng, queryVars)
          % postQuery = sum_h p(Query,h)
          if any(ismember(queryVars,eng.visVars))
            warning('VarElimInfEng:alreadyConditioned','You have already conditioned on one or more of your query variables.');
          end
          % factors involving continuous, unobserved nodes
          % We exploit the fact that MixtureDist returns a TabularFactor of 1
          % if both the discrete indicator and the (possibly cts) child are
          % hidden
          %postQuery = varElimCts(model, Tfac, cfacs, queryVars)
          % Find the hidden cts child nodes
          cfacs = (cellfun(@(x)isequal(pmf(x),1),eng.Tfac));  % factors involving continuous, unobserved nodes
           cfacs = find(cfacs);
          if eng.verbose, fprintf('preparing VarElim\n'); end
          for i=1:length(cfacs)
            dom = eng.Tfac{cfacs(i)}.domain;
            child = dom(end);
            if ismember(child, queryVars)
              error('cannot query the child of a latent mixture node')
            end
          end
          elim = setdiffPMTK(setdiffPMTK(eng.domain(eng.ordering),queryVars),eng.visVars);
           if eng.verbose, fprintf('running VarElim\n'); end
          postQuery = VarElimInfEng.variableElimination(eng.Tfac,elim); % real work happens here
          [postQuery,Z] = normalizeFactor(postQuery);
           if eng.verbose, fprintf('done\n'); end
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
                tau = marginalize(psi,setdiffPMTK(psi.domain,variable)); % marginalize out the elimination variable
                newFactors = {factors{not(inscope)},tau};
            end
            
            
        end
        
        
    end
    
    
    
   
end