classdef UgmTabularDist < UgmDist 
    % undirected graphical model with tabular potentials
    
    properties
        %G; infMethod;
        factors;
    end
    
    %%  Main methods
    methods
        function obj = UgmTabularDist(varargin)
            % UgmTabularDist(...)
            % 'G' - graph structure
            % 'factors' - cell array of tabularFactor
            % 'infMethod' - {'varElim', 'enum'}
            % nstates(j) - number of states for node j
            if(nargin == 0);return;end
            [G, obj.factors, obj.infEng, obj.nstates] = process_options(varargin, ...
                'G', [], 'factors', [], 'infEng', JtreeInfEng(), 'nstates', []);
            if isempty(G)
                % infer graph topology from factors
                d = length(obj.nstates);
                G = zeros(d,d);
                n = length(obj.factors);
                for j=1:n
                    dom = obj.factors{j}.domain;
                    G(dom,dom)=1;
                end
                G = setdiag(G,0);
            end
            if isa(G, 'double'), G = UndirectedGraph(G); end
            obj.G = G;
            obj.domain = 1:nnodes(G);
            obj.discreteNodes = 1:nnodes(G);
        end
       
        
        function [Tfacs,nstates] = convertToTabularFactors(obj,visVars,visVals)
            if 0 % nargin == 1 || isempty(visVars)
              % we want to comptue nstates from factors in case it is not
              % specified
                Tfacs = obj.factors;
                nstates = obj.nstates;
                return;
            end
            d = length(obj.factors);
            Tfacs = obj.factors;
            for j=1:d
                include = ismember(visVars,Tfacs{j}.domain);
                if(~isempty(include) && any(include))
                    Tfacs{j} = slice(Tfacs{j},visVars(include),visVals(include));
                end
                nstates(j) = Tfacs{j}.sizes(end);
            end
        end
        
        function Tfac = convertToJointTabularFactor(obj,visVars,visVals)
          % Represent the joint distribution as a single large factor
          if nargin < 3
            visVars = []; visVals = [];
          end
          Tfac = TabularFactor.multiplyFactors(convertToTabularFactors(obj,visVars,visVals));
        end
        
        function fc = makeFullConditionals(obj, visVars, visVals)
            d = ndimensions(obj);
            if nargin < 2
                % Sample from the unconditional distribution
                visVars = []; visVals = [];
            end
            V = visVars; H = setdiffPMTK(1:d, V);
            x = zeros(1,d); x(V) = visVals;
            fc = cell(length(H),1);
            for i=1:length(H)
                Tfac = jointMarkovBlanket(obj, i);
                fc{i} = @(xh) fullCond(obj, xh, i, H, x, Tfac);
            end
        end
        
        function p = fullCond(obj, xh, i, H, x, Tfac) %#ok
            assert(length(xh)==length(H))
            % xh is current state of Gibbs sampler
            x(H) = xh; % insert sampled hidden values into hidden slot
            %x(i) = []; % remove value for i'th node, which will be sampled
            visVars = setdiffPMTK(Tfac.domain, i);
            visValues = x(visVars);
            p = normalizeFactor(slice(Tfac, visVars, visValues));
        end
        
        function Tfac = jointMarkovBlanket(obj, node)
            Tfacs = {};
            for j=1:length(obj.factors)
                inter = intersectPMTK(obj.factors{j}.domain, node);
                if ~isempty(inter)
                    Tfacs{end+1} = obj.factors{j}; %#ok
                end
            end
            Tfac = TabularFactor.multiplyFactors(Tfacs);
        end
        
        function xinit = mcmcInitSample(model, visVars, visVals) %#ok that we ignore visVals
            if nargin < 2
                visVars = []; visVals = [];
            end
            % Ideally we would draw an initial sample conditional on the
            % observed data.
            % Instead we sample from the prior.
            % However, even this is hard in general.
            % So we just sample a random vector from the uniform distribution
            % (this might be inconsistent with hard constraints!!)
            domain = 1:ndimensions(model);
            hidVars = setdiffPMTK(domain, visVars);
            V = lookupIndices(visVars, domain);
            H = lookupIndices(hidVars, domain);
            sz = model.nstates(H);
            K = prod(sz);
            p = (1/K)*ones(1,length(H));
            xinit = ind2subv(sz, sample(p,1));
            assert(length(H)==length(xinit))
        end
        
        
        
        
    end
    
    
end