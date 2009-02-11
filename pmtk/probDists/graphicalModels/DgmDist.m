classdef DgmDist < GmDist
  % directed graphical model
  
  properties
    CPDs;
    %infEng;
    %infMethod;
    % G field is in parent class
  end

  
  %%  Main methods
  methods
      function obj = DgmDist(G, varargin)
          if(nargin == 0);G = [];end
          if isa(G,'double'), G=Dag(G); end
          obj.G = G;
          [obj.CPDs, obj.infEng,obj.domain]= process_options(...
              varargin, 'CPDs', [], 'infEng', [],'domain',[]);
          %if ~isempty(CPDs) && ~isempty(infMethod)
          %  obj = initInfEng(obj);
          %end
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
        %X = sample(obj.infEng, n); % not necessary to use joint!!
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
       %dgm = initInfEng(dgm);
   end
   
    function M = convertToUgm(obj)
      [Tfacs, nstates] = convertToTabularFactors(obj);
      M = UgmTabularDist('factors', Tfacs, 'nstates', nstates);
    end
    
    function [Tfacs, nstates] = convertToTabularFactors(obj,visVars,visVals)
        if(nargin < 3)
           visVars = [];  visVals = {};
        end
        d = length(obj.CPDs);
        Tfacs = cell(1,d);
        nstates = zeros(1,d);
        for j=1:d
            dom = [parents(obj.G, j), j];
            include = ismember(visVars,dom);
            if(isempty(include) || ~any(include))
                 Tfacs{j} = convertToTabularFactor(obj.CPDs{j}, dom, [],{});
            else
                 Tfacs{j} = convertToTabularFactor(obj.CPDs{j}, dom, visVars(include) , visVals(include));
            end
            nstates(j) = Tfacs{j}.sizes(end);
        end
    end
    
    function Tfac = convertToTabularFactor(obj,visVars,visVals)
      % Represent the joint distribution as a single large factor
      if nargin < 3
          visVars = []; visVals = [];
      end
      Tfac = TabularFactor.multiplyFactors(convertToTabularFactors(obj,visVars,visVals));
    end
    
    function nodeDist = extractLocalDistribution(obj,var)
       nodeDist = obj.CPDs{var};
    end
    
    function [mu,Sigma,domain] = convertToMvnDist(dgm)
        % Koller and Friedman p233
        d = nnodes(dgm.G);
        mu = zeros(d,1);
        Sigma = zeros(d,d);
        for j=1:d
            if isa(dgm.CPDs{j}, 'ConstDist') % node was set by intervention
                b0 = 0; w = 0; sigma2 = 0;
            else
                b0 = dgm.CPDs{j}.w0;
                w = dgm.CPDs{j}.w;
                sigma2 = dgm.CPDs{j}.v;
            end
            pred = parents(dgm.G,j);
            beta = zeros(j-1,1);
            beta(pred) = w;
            mu(j) = b0 + beta'*mu(1:j-1);
            Sigma(j,j) = sigma2 + beta'*Sigma(1:j-1,1:j-1)*beta;
            s = Sigma(1:j-1,1:j-1)*beta;
            Sigma(1:j-1,j) = s;
            Sigma(j,1:j-1) = s';
        end
        %dist = MvnDist(mu,Sigma);
        domain = 1:length(mu);
    end
    
  end



end


