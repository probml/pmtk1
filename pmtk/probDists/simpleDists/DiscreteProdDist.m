classdef DiscreteProdDist  < ParamJointDist
% This class represents a product of Multinoullis


  properties
    %T; %num states * num dimns
    params;
    prior; % DirichletDist or 'none'
    priorStrength;
    support;
    
    % for inference
    visVars; visVals;
  end
  
  
  methods
    function obj = DiscreteProdDist(varargin) 
      % obj = DiscreteProdDist(T, nstates, ndims, support, prior, priorStrength)
      % p(x|theta) = prod_{j=1}^d prod_{k=1}^{K} T(k,j)^{I(x(j)=k}
      % where d is number of dimensions of x, and K is the number
      % of states.
      % 'T' - T is K*d, for K states and d distributions.
      %    Each *column* of T represents a different discrete distribution. 
      % 'support' - Support is a set of K numbers, defining the domain.
      % nstates - defines support to be {1,2,...,nstates}
      % Each distribution has the same support.
      % 'prior' - NoPrior or 'dirichlet' or DirichletDist.
      % Same prior is used for each distribution.
      if nargin == 0; return ; end % must be able to call the constructor with no args...
      [T, nstates, ndims, support, prior, obj.priorStrength, obj.productDist] = ...
        processArgs(varargin, ...
        '-T', [], '-nstates', [], '-support', [], '-ndims', [], ...
        '-prior', NoPrior, ...
        '-priorStrength', 0, '-productDist', false);
      if isempty(T)
        d = ndims; K = nstates;
        if isempty(d) || isempty(K)
          error('must specify d,K or T')
        end
        obj = mkRndParams(obj, d, K);
      end
      if isempty(support) 
        [nstates] = size(T,1);
        support = 1:nstates;
      end
      %if isempty(support), error('must specify support or nstates or T'); end
      if(~approxeq(normalize(T,1),T))
         error('Each column must sum to one'); 
      end
      obj.params.T = T;
      obj.support = support;
      obj.prior = prior;
    end

  
    function L = logprob(obj,X)
      % L(i) = sum_j log p(X(i,j) | params(j)) 
      X = canonizeLabels(X,obj.support);
      n = size(X,1);
      T = obj.params.T;
      d = size(T,2);
      Lij = zeros(n,d);
      for j=1:d
        Lij(:,j) = log(T(X(:,j),j));
      end
      L = sum(Lij,2);
    end

    function L = logprior(model)
      L = logprob(model.prior, model.params.T);
    end
    
    
    function obj = mkRndParams(obj,d,K)
      if nargin < 2
        T = obj.params.T;
        [d,K]  = size(T);
      end
      obj.T = normalize(rand(K,d),1);
    end
    
    
    
    %% Fitting
    function model = fit(model,varargin)
      % model = fit(model, data, suffStat)
      % data(i,j) is value of case i, variable j (an integer in model.support)
      % suffStat.counts is a K*d matrix
      [X, SS] = processArgs(varargin, ...
        '-data', [], ...
        '-suffStat', []);
      if isa(X,'DataTable'), X = X.X; end
      if ~isempty(X) && any(isnan(X))
        model = fitMissingData(model,X);
        return;
      end
      if isempty(SS), SS = mkSuffStat(model, X); end
      d = size(SS.counts,2);
      if ~isa(model.prior, 'ProbDist'), model = initPrior(model); end
      switch class(model.prior)
        case 'DirichletDist'
          pseudoCounts = repmat(model.prior.alpha(:),1,d);
          model.T = normalize(SS.counts + pseudoCounts -1, 1);
        otherwise
          error('unknown prior ')
      end % switch prior
    end % fit
  
    
    
    %% Inference
    
    % Impute is inherited from ParamJointDist
    % The other methods are implemented here directly,
    % withotu calling out to an inference engine
    
    function model = condition(model, visVars, visValues)
      % enter evidence that visVars=visValues
      if nargin < 2
        visVars = []; visValues = [];
      end
      model.visVars = visVars;
      model.visVals = visValues;
     % Just memorize the data!
    end
    
    function [postQuery,model] = marginal(model, Q)
      % p(Q|evidence entered)
      % Joint distribution is itself factored
      dQ = length(Q);
      T  = model.params.T;
      [K] = size(T,1);
      TQ = zeros(K,dQ);
      Qndx = lookupIndices(Q, model.visVars);
      for j=1:dQ
        q = Q(j);
        if ismember(q, model.visVars)
          TQ(:,j) = model.visVals(Qndx(j));
          % delta function on the observed value
        else
          TQ(:,j) = T(:,q);
          % prior distribution for this dimension
        end
      end
      postQuery = DiscreteProdDist('-T',TQ);
    end
    
    function X = sample(obj, n)
      % x(i,j) is an integer in the support for i=1:n, j=1:d
      if nargin < 2, n = 1; end
      T = obj.params.T;
      [K,d] = size(T);
      X = zeros(n, d);
      V = obj.visVars;
      % This is wrong if the domain is not labeled as 1:d
      H = setdiffPMTK(1:d, V); 
      % replicate observed values
      for jV=1:length(V)
        j = V(jV);
        X(:,j) = obj.visVars(jV)*ones(n,1);
      end
      % sample hidden variables
      for jH=1:length(H)
        j = H(jH);
        p = T(:,j); cdf = cumsum(p);
        [dum, y] = histc(rand(n,1),[0 ;cdf]);
        X(:,j) = obj.support(y);
      end
    end
    
     function x = mode(obj)
       % x(j) is most probable value of j, given any evidence
      T = obj.params.T;
      [K,d] = size(T);
      x = zeros(1, d);
      V = obj.visVars;
      % This is wrong if the domain is not labeled as 1:d
      H = setdiffPMTK(1:d, V); 
      % replicate observed values
      for jV=1:length(V)
        j = V(jV);
        x(j) = obj.visVars(jV);
      end
      % compute mode of each hidden variable
      for jH=1:length(H)
        j = H(jH);
        x(j) = obj.support(argmax(T(:,j)));
      end
     end
    
    
  end % methods

 methods(Access = 'protected')
   
    function model = fitMissingData(model,X)
      % We fit each dimension separately.
      % We just omit the missing data from each column.
      if ~isa(model.prior, 'ProbDist'), model = initPrior(model); end
      if ~isa(model.prior, 'DirichletDist')
        error('can only handle dirichlet prior')
      end
      [n,d] = size(X);
      pseudoCounts = model.prior.alpha(:);
      K = length(obj.support);
      T = zeros(K,d);
      for j=1:d
        Xj = X(:,j);
        Xj = Xj(~isnan(Xj));
        counts = colvec(histc(Xj, obj.support));
        T(:,j) =  normalize(counts + pseudoCounts -1, 1);
      end
    end % fitMissingData
   
    
   function model = initPrior(model) 
     if isempty(model.prior), model.prior = 'none'; end
     %if ~ischar(model.prior), return; end
     T = model.params.T;
     [K] = size(T,1);
     switch lower(model.prior)
       case 'none',
         alpha = ones(K,1);
       case 'dirichlet'
         alpha = model.priorStrength*ones(K,1);
       case 'jeffreys'
         alpha = (1/K)*ones(K,1);
       otherwise
         error(['unknown prior ' model.prior])
     end
     model.prior = DirichletDist(alpha);
   end
   
     
   function SS = mkSuffStat(obj, X,weights)
      K = nstates(obj);
      d = size(X,2);
      counts = zeros(K, d);
      X = double(full(X));
      %X = canonizeLabels(X, obj.support);
      %S = length(obj.support);
      if(nargin < 3)
        for j=1:d
          counts(:,j) = colvec(histc(X(:,j), obj.support));
        end
      else  %weighted SS
        if size(weights,2) == 1
          weights = repmat(weights,1,d);
        end
        for j=1:d
          for s=1:K % can't we vectorize this more?
            counts(s,j) = sum(weights(X(:,j) == obj.support(s),j));
          end
        end
      end
      SS.counts = counts;
   end
    
    
 end % protected methods
  
 methods(Static = true)
   function testClass()
     d = 4; K = 3;
     m = DiscreteProdDist('-d', d,'-K',K);
   end
   
 end % static methods
 
end


