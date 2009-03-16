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
      N = nnodes(G);
      [obj.CPDs, obj.infMethod, obj.domain, ...
        obj.discreteNodes, obj.nstates]= process_options(...
        varargin, 'CPDs', [], 'infMethod', JtreeInfEng(), 'domain',[], ...
        'discreteNodes', 1:N, 'nstates', ones(1,N));
      if isempty(obj.domain)
        obj.domain = 1:nnodes(G);
      end
      if ~isempty(obj.CPDs) && ~isempty(G)
        assert(length(obj.CPDs) == nnodes(G)) % not true if there is param tying
      end
      if ~isempty(obj.CPDs)
        obj.discreteNodes = find(cellfun(@isDiscrete, obj.CPDs));
        obj.ctsNodes = setdiffPMTK(1:N, obj.discreteNodes);
        obj.nstates = cellfun(@nstates, obj.CPDs);
      end
    end

      function d = ndimensions(obj)
          d = nnodes(obj.G);
      end
      
      function M = convertToUgm(obj, visVars, visVals)
        if(nargin < 2), visVars = [];  visVals = []; end
        [Tfacs, nstates] = convertToTabularFactors(obj, visVars, visVals);
        M = UgmTabularDist('factors', Tfacs, 'nstates', nstates);
      end
    
      function [Tfacs, nstates] = convertToTabularFactors(obj,visVars,visVals)
        if(nargin < 2), visVars = [];  visVals = {}; end
        d = nnodes(obj.G);
        data = sparse(1,d);
        data(visVars) = visVals;
        visible = false(1,d);
        visible(visVars) = true;
        Tfacs = cell(1,d);
        nstates = zeros(1,d);
        for j=1:d
          %{
      dom = [parents(obj.G, j), j];
      include = ismember(visVars,dom);
      if(isempty(include) || ~any(include))
        Tfacs{j} = convertToTabularFactor(obj.CPDs{j}, dom, [],{});
      else
        Tfacs{j} = convertToTabularFactor(obj.CPDs{j}, dom, visVars(include) , visVals(include));
      end
          %}
          ps = parents(obj.G, j);
          ctsParents = intersectPMTK(ps, obj.ctsNodes);
          dParents = intersectPMTK(ps, obj.discreteNodes);
          child = j;
          Tfacs{j} = convertToTabularFactor(obj.CPDs{j}, child, ctsParents, dParents, visible, data, obj.nstates);
          nstates(j) = Tfacs{j}.sizes(end);
          %assert(nstates(j)==obj.nstates(j)); % nstates is after seeing
          %evidence
        end
      end
    
    function Tfac = convertToJointTabularFactor(obj,visVars,visVals)
      % Represent the joint distribution as a single large factor
      if nargin < 3
        visVars = []; visVals = [];
      end
      Tfac = TabularFactor.multiplyFactors(convertToTabularFactors(obj,visVars,visVals));
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
    end

%    function nodeDist = extractLocalDistribution(obj,var)
%    % Used to extract the distribution over a discrete parent of a 
%    % continuous node to get the mixing weights for the induced MixtureDist.
%        nodeDist = obj.CPDs{var};
%     end
  
    
    
  end



end


