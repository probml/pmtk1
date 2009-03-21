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
            N = nnodes(model.G);
            visible = false(1,N); visible(visVars) = true;
            hidCts = model.ctsNodes(~visible(model.ctsNodes));
            if any(arrayfun(@(i) ~isleaf(model.G, i), hidCts))
                error('VarElimInfEng requires all hidden cts nodes to be leaves')
            end
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
            if any(ismember(queryVars,eng.model.ctsNodes))
               [postQuery,eng,Z] = handleContinuousQuery(eng,queryVars); return;
            end
            if eng.verbose, fprintf('preparing VarElim\n'); end
            elim = setdiffPMTK(setdiffPMTK(eng.domain(eng.ordering),queryVars),eng.visVars);
            if eng.verbose, fprintf('running VarElim\n'); end
            postQuery = VarElimInfEng.variableElimination(eng.Tfac,elim); % real work happens here
            [postQuery,Z] = normalizeFactor(postQuery);
            if eng.verbose, fprintf('done\n'); end
                
        end
        
        
    end % methods
    
    
    methods(Access = 'protected')
       
        function [postQuery,eng,Z] = handleContinuousQuery(eng,queryVars)
        % Handle queries on unobserved, continuous leaf nodes with discrete
        % parents in directed models. The return type is a MixtureDist, (or
        % subclass thereof), whose mixing weights are given by the
        % posterior marginal of the discrete parent.
            if(numel(queryVars) ~= 1)
               error('you can only query a single continuous node at a time'); 
            end
            if ~isempty(children(eng.model.G.adjMat,find(queryVars == eng.domain)))
               error('unobserved, continuous, query nodes must be leaves'); 
            end
            if ~isdirected(eng.model)
               error('querying unobsesrved, continous nodes currently only supported in directed models'); 
            end
            parent = eng.domain(parents(eng.model.G.adjMat,find(queryVars == eng.domain)));
            if numel(parent) > 1
               error('a continuous node can have only one discrete parent');
            end
            postQuery = extractLocalDistribution(eng.model,queryVars);
            SS.counts = pmf(marginal(eng,parent));
            assert(isa(postQuery,'MixtureDist'));
            postQuery.mixingWeights = fit(postQuery.mixingWeights,'suffStat',SS);
            Z = 1;
        end
        
    end
    
    
    methods(Static = true)
        
        function margFactor = variableElimination(factors,elimOrdering)
            % Perform sum-product variable elimination
            % See Koller & Friedman algorithm 9.1 pg 273
            for i=1:numel(elimOrdering)
                factors = eliminate(elimOrdering(i),factors);
            end
            margFactor = TabularFactor.multiplyFactors(factors);
            
            function newFactors = eliminate(variable,factors)
            % eliminate a single variable
                inscope = cellfun(@(fac)ismember(variable,fac.domain),factors); % inscope(f) is true iff the variable is in the scope of factors{f}
                psi = TabularFactor.multiplyFactors(factors(inscope));
                tau = marginalize(psi,setdiffPMTK(psi.domain,variable));        % marginalize out the elimination variable
                newFactors = {factors{not(inscope)},tau};
            end
            
            
        end
        
        
    end % static methods
    
    
    
    
end