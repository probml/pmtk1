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
            map = @(x)canonizeLabels(x,model.domain); % maps domain to 1:d, inverse map is model.domain(x)
            % barren nodes have already been removed by GmDist.marginal so
            % we just need to check that all continuous nodes are leaves and
            % error if not.           
            if any(arrayfun(@(n) ~isleaf(model.G.adjMat, n), map(model.ctsNodes)))
                error('Unobserved, continuous, query nodes, must have no observed children.');
            end
            [eng.Tfac,nstates] = convertToTabularFactors(model,visVars,visVals);
            if isempty(eng.ordering)
                if(model.G.directed)
                    moralGraph = moralize(model.G);
                else
                    moralGraph = model.G;
                end
                eng.ordering = model.domain(best_first_elim_order(moralGraph.adjMat,nstates)); %w.r.t. model.domain, not necessarily 1:d
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
            elim = setdiff(setdiff(eng.ordering,queryVars),eng.visVars);
            if eng.verbose, fprintf('running VarElim\n'); end
            postQuery = VarElimInfEng.variableElimination(eng.Tfac,elim); % real work happens here
            [postQuery,Z] = normalizeFactor(postQuery);
            if eng.verbose, fprintf('done\n'); end
                
        end
        
        function X = sample(eng,n)
           error('Sampling is not implemented for VarElimInfEng, use JtreeInfEng instead');
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
            if ~isdirected(eng.model)
               error('querying unobsesrved, continuous nodes currently only supported in directed models'); 
            end
            parent = eng.domain(parents(eng.model.G.adjMat,canonizeLabels(queryVars,eng.domain)));
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