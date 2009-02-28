classdef DgmDist < GmDist
  % directed graphical model
  
  properties
    CPDs;
    % infMethod in GM class
    % G field is in parent class
  end

  
  %%  Main methods
  methods
    function obj = DgmDist(G, varargin)
      if(nargin == 0);G = [];end
      if isa(G,'double'), G=Dag(G); end
      obj.G = G;
      [obj.CPDs, obj.infMethod, obj.domain]= process_options(...
        varargin, 'CPDs', [], 'infMethod', 'varElim', 'domain',[]);
      if isempty(obj.domain)
        obj.domain = 1:numel(obj.CPDs);
      end
    end

      function d = ndimensions(obj)
          d = nnodes(obj.G);
      end
      
    function obj = fit(obj, X, varargin)
        % We fit each CPD separately assuming no missing data
        % and no param tying
        assert(~any(isnan(X(:))));
        [n d] = size(X);
        assert(d == ndimensions(obj));
        [clampedCPDs, interventionMask] = process_options(varargin, ...
            'clampedCPDs', false(1,d), 'interventionMask', false(n,d));
        for j=find(~clampedCPDs)
            pa = parents(obj.G, j);
            %fprintf('fit node %d\n', j);
            % Only including training cases that were not set by intervention
            ndx = find(~interventionMask(:,j));
            obj.CPDs{j} = fit(obj.CPDs{j}, 'X', X(ndx, pa), 'y', X(ndx,j));
        end
        %obj = initInfEng(obj);
    end
    
    function X = sample(obj, n)
        % X(i,:) = i'th forwards (ancestral) sample
        d = ndimensions(obj);
        X = zeros(n,d);
        for j=1:d
            pa = parents(obj.G, j);
            X(:,j) = sample(obj.CPDs{j}, X(:,pa), n);
        end
    end
    
   
    %{
    function postQuery = varElimCts(model, Tfac, cfacs, queryVars)
      cdom = cell2mat(cellfuncell(@(x)rowvec(sube(subd(x,'domain'),2)),Tfac(cfacs)));
      % domain of unobserved continuous nodes
      I = intersect(cdom,queryVars);
      if ~isempty(I)
        if numel(queryVars) > numel(I)
          error('Cannot represent the joint over mixed node types.');
        elseif numel(I) > 1
          error('Cannot represent the joint over two or more unobserved continuous nodes.');
        else
          postQuery = extractLocalDistribution(model,I);
          assert( isa(postQuery,'MixtureDist'));
          par = parents(model.G, I);
          assert(numel(par) == 1);
          SS.counts = pmf(marginal(model,par));
          postQuery.mixingWeights = fit(postQuery.mixingWeights,'suffStat',SS);
          return;
        end
      end
    end
%}
    
    %{
    function postQuery = predict(obj, visVars, visVals, queryVars, varargin)
      % sum_h p(Q,h|V=v)
      % 'interventionVector'(j) = 1 if node j set by intervention
      % Needs to be vectorized...
      [interventionVector] = process_options(varargin, ...
        'interventionVector', []);
      if ~isempty(interventionVector)
        obj = performIntervention(obj, visVars, visVals, interventionVector);
      end
     postQuery = predict(obj.infEng, obj, visVars, visVals, queryVars);
    end
 %}
    
    function obj = performIntervention(obj, visVars, visVals, interventionVector)
        % Perform Pearl's "surgical intervention" on nodes specified by
        % intervention vector. We modify the specified CPDs and
        % reinitialize the inference engine.
        for j=interventionVector(:)'
            ndx = (visVars==j);
            obj.CPDs{j} = ConstDist(visVals(ndx));
        end
        %obj = initInfEng(obj);
    end
     
    function L = logprob(obj, X, varargin)
        % L(i) = log p(X(i,:) | params)
        [n d] = size(X);
        assert(~any(isnan(X(:))));
        [interventionMask] =  process_options(varargin, ...
            'interventionMask', false(n,d));
        L = zeros(n,d);
        for j=1:d
            pa = parents(obj.G, j);
            % Only including  cases that were not set by intervention
            ndx = find(~interventionMask(:,j));
            L(ndx,j) = logprob(obj.CPDs{j}, X(ndx, pa), X(ndx, j));
        end
        L = sum(L,2);
    end

    function L = logmarglik(obj, X, varargin)
        % L = log p(X)
        [n d] = size(X);
        assert(~any(isnan(X(:))));
        [interventionMask] =  process_options(varargin, ...
            'interventionMask', false(n,d));
        L = zeros(1,d);
        for j=1:d
            pa = parents(obj.G, j);
            % Only including  cases that were not set by intervention
            ndx = find(~interventionMask(:,j));
            L(j) = logmarglik(obj.CPDs{j}, X(ndx, pa), X(ndx, j));
        end
        L = sum(L);
    end
    
   
    %{
   function dgm = mkRndParams(dgm, varargin)
       d = ndimensions(dgm);
       [CPDtype, arity] = process_options(varargin, ...
           'CPDtype', 'TabularCPD', 'arity', 2*ones(1,d));
       for j=1:d
           pa = parents(dgm.G, j);
           switch lower(CPDtype)
               case 'lingausscpd',
                   q  = length(pa);
                   dgm.CPDs{j} = LinGaussCPD(randn(q,1), randn(1,1), rand(1,1));
               case 'tabularcpd'
                   q  = length(pa);
                   dgm.CPDs{j} = TabularCPD(mkStochastic(rand(arity([pa j]))));
               otherwise
                   error(['unknown type ' CPDtype])
           end
       end
       %dgm = initInfEng(dgm);
   end
   %}
    
   
    
    function nodeDist = extractLocalDistribution(obj,var)
      error('what is this for?')
       nodeDist = obj.CPDs{var};
    end
    
    
  end



end


