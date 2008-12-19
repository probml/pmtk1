classdef DgmDist < GmDist
  % directed graphical model
  
  properties
    CPDs;
    infEng;
    infMethod;
    % G field is in parent class
  end

  
  %%  Main methods
  methods
    function obj = DgmDist(G, varargin)
      if isa(G,'double'), G=Dag(G); end
      obj.G = G;
      [CPDs, infMethod]= process_options(...
        varargin, 'CPDs', [], 'infMethod',[]);
      obj.CPDs = CPDs;
      obj.infMethod = infMethod;
      if ~isempty(CPDs) && ~isempty(infMethod)
        obj = initInfEng(obj);
      end
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
      obj = initInfEng(obj);
    end
    
    function X = sample(obj, n)
      % X(i,:) = i'th forwards (ancestral) sample
      X = sample(obj.infEng, n); % not necessary to use joint!!
    end
    
    function postQuery = marginal(obj, queryVars)
      postQuery = marginal(obj.infEng, obj, queryVars);
    end
    
    function postQuery = predict(obj, visVars, visVals, queryVars, varargin)
      % sum_h p(Q,h|V=v)
      % Needs to be vectorized...
      [interventionVector] = process_options(varargin, ...
        'interventionVector', []);
      if ~isempty(interventionVector)
        obj = performIntervention(obj, visVars, visVals, interventionVector);
      end
      postQuery = predict(obj.infEng, visVars, visVals, queryVars);
    end
     
    function obj = performIntervention(obj, visVars, visVals, interventionVector)
      % Perform Pearl's "surgical intervention" on nodes specified by
      % intervention vector. We modify the specified CPDs and 
      % reinitialize the inference engine.
      for j=interventionVector(:)'
        ndx = (visVars==j);
        obj.CPDs{j} = ConstDist(visVals(ndx));
      end
      obj = initInfEng(obj);
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
    
    function obj = initInfEng(obj, infMethod)
      % "Freeze" the CPDs and graph structure and compile into an engine
      if nargin < 2, infMethod = obj.infMethod; end
      if isempty(infMethod), return; end % silently return
      switch lower(infMethod)
        case 'enumeration'
          T =  dgmDiscreteToTable(obj);
          obj.infEng = EnumInfEng(T);
        case 'gauss'
          [mu, Sigma] = dgmGaussToMvn(obj);
          obj.infEng = GaussInfEng(mu, Sigma);
        otherwise
          error(['unrecognized method ' infMethod])
      end
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
      dgm = initInfEng(dgm);
    end
    
    function T = dgmDiscreteToTable(obj)
      d = length(obj.CPDs);
      Tfacs = cell(1,d);
      for j=1:d
        if isa(obj.CPDs{j}, 'ConstDist') % node was set by intervention
          sz = mysize(obj.CPDs{j}.T);
          ssz = sz(end);
          tmp = zeros(1,ssz);
          tmp(obj.CPDs{j}.point) = 1; % delta function at set value
          Tfacs{j} = TabularFactor(tmp, j);
        else % CPDs over discrete nodes can always be converted to tables...
          dom = [parents(obj.G, j), j];
          Tfacs{j} = convertToTabularFactor(obj.CPDs{j}, dom);
        end
      end
      Tfac = TabularFactor.multiplyFactors(Tfacs);
      T = Tfac.T;
      %T = TableJointDist(Tfac.T);
    end
    
    function [mu, Sigma] = dgmGaussToMvn(dgm)
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
      %mvn = MvnDist(mu,Sigma);
    end
    
  end


  methods(Static = true)
   
    
  end

end


