classdef MvnDist < ParamDist 
  % multivariate normal p(X|mu,Sigma) 
  
  properties
    mu;
    Sigma;
    domain;
  end
  
  %% main methods
  methods
    function m = MvnDist(mu, Sigma, domain)
      % MvnDist(mu, Sigma)
      % mu can be a matrix or a pdf, eg. 
      % MvnDist(MvnInvWishDist(...), [])
      if nargin == 0
        mu = []; Sigma = [];
      end
      if nargin < 3, domain = 1:length(mu); end
      m.mu  = mu;
      m.Sigma = Sigma;
      m.domain = domain;
    end

    function params = getModelParams(obj)
      params = {obj.mu, obj.Sigma};
    end
    
    function objS = convertToScalarDist(obj)
      if ndimensions(obj) ~= 1, error('cannot convert to scalarDst'); end
      objS = GaussDist(obj.mu, obj.Sigma);
    end
    
    function obj = mkRndParams(obj, d)
      if nargin < 2, d = ndimensions(obj); end
      obj.mu = randn(d,1);
      obj.Sigma = randpd(d);
    end
    
    function d = ndimensions(m)
      if isa(m.mu, 'double')
        d = length(m.mu);
      else
        d = ndimensions(m.mu);
      end
    end

    function logZ = lognormconst(obj)
      d = ndimensions(obj);
      logZ = (d/2)*log(2*pi) + 0.5*logdet(obj.Sigma);
    end
    
    function L = logprob(obj, X, normalize)
      % L(i) = log p(X(i,:) | params)
      if nargin < 3, normalize = true; end
      mu = obj.mu(:)'; % ensure row vector
      if length(mu)==1
        X = X(:); % ensure column vector
      end
      [N d] = size(X);
      if length(mu) ~= d
        error('X should be N x d')
      end
      %if statsToolboxInstalled
      %  L1 = log(mvnpdf(X, obj.mu, obj.Sigma));
      M = repmat(mu, N, 1); % replicate the mean across rows
      if obj.Sigma==0
        L = repmat(NaN,N,1);
      else
        mahal = sum(((X-M)*inv(obj.Sigma)).*(X-M),2);
        if normalize
          L = -0.5*mahal - lognormconst(obj);
        else
          L = -0.5*mahal;
        end
      end
      %assert(approxeq(L,L1))
    end
    
    %{
    function h=plotContour2d(obj, varargin)
      % Plot an ellipse representing the 95% contour of a Gaussian
      % eg figure; plotContour2d(mvnDist([0 0], [2 1; 1 1]))
      checkParamsAreConst(obj)
      if ndimensions(obj) ~= 2
        error('only works for 2d')
      end
      h = gaussPlot2d(obj.mu, obj.Sigma);
    end
     %}
  
    function mu = mean(m)
      checkParamsAreConst(m)
      mu = m.mu;
    end

    function mu = mode(m)
      mu = mean(m);
    end

    function C = cov(m)
      checkParamsAreConst(m)
      C = m.Sigma;
    end
  
    function v = var(obj)
      v = diag(cov(obj));
    end
    
    
    function samples = sample(obj,n)
      % Sample n times from this distribution: samples is of size
      % nsamples-by-ndimensions
      checkParamsAreConst(obj);
      if(nargin < 2), n = 1; end;
      A = chol(obj.Sigma,'lower');
      Z = randn(length(obj.mu),n);
      samples = bsxfun(@plus,obj.mu(:), A*Z)';
    end
    
     function [postQuery] = marginal(obj, queryVars)
       % prob = sum_h p(Query,h)
       checkParamsAreConst(obj);
       Q = lookupIndices(queryVars, obj.domain);
       postQuery = MvnDist(obj.mu(Q), obj.Sigma(Q,Q), queryVars);
     end
     
     function prob = predict(obj, visVars, visValues, queryVars)
      %prob =  sum_h p(query,h|visVars=visValues)
      H = mysetdiff(obj.domain, visVars);
      checkParamsAreConst(obj);
      [muHgivenV, SigmaHgivenV] = gaussianConditioning(...
        obj.mu, obj.Sigma, visVars, visValues); 
      prob = MvnDist(muHgivenV, SigmaHgivenV, H);
      if nargin >= 4
        prob = marginal(prob, queryVars);
      end
     end
   
      function Xc = impute(obj, X)
       % Fill in NaN entries of X using posterior mode on each row
       checkParamsAreConst(obj);
       [n d] = size(X);
       Xc = X;
       for i=1:n
         hidNodes = find(isnan(X(i,:)));
         if isempty(hidNodes), continue, end;
         visNodes = find(~isnan(X(i,:)));
         visValues = X(i,visNodes);
         postH = predict(obj, visNodes, visValues);
         mu = mode(postH);
         Xc(i,hidNodes) = mu(:)';
       end
      end
      
      
      function fc = makeFullConditionals(obj, visVars, visVals)
        d = length(obj.mu);
        if nargin < 2
          % Sample from the unconditional distribution
          visVars = []; visVals = [];
        end
        V = visVars; H = mysetdiff(1:d, V);
        x = zeros(1,d); x(V) = visVals;
        for i=1:length(H)
          fc{i} = @(xh) fullCond(obj, xh, i, H, x);
        end
      end
      
      function p = fullCond(obj, xh, i, H, x)
        x(H) = xh; % insert sampled hidden values into hidden slot
        x(i) = []; % remove value for i'th node, which will be sampled
        d = length(obj.mu);
        dom = 1:d; dom(i) = []; % specify that all nodes are observed except i
        [muAgivenB, SigmaAgivenB] = gaussianConditioning(obj.mu, obj.Sigma, dom, x);
        %xi = normrnd(muAgivenB, sqrt(SigmaAgivenB));
        p = GaussDist(muAgivenB, SigmaAgivenB);
      end
    
     
      
    function obj = fit(obj,varargin)
    % Fit the distribution via the specified method
    %
    % FORMAT: 
    %
    %  obj = fit(obj,'name1',val1,'name2',val2,...);
    %
    % INPUT:
    %
    % 'data'     -        data(i,:) is case i
    % 'suffStat' -        the sufficient statistics of the data made via
    %                     SS = MvnDist.mkSuffStat(data). If not specified, this
    %                     is automatically calculated.
    %
    % 'method'   -        If either obj.mu or obj.Sigma is an object, 
    %                     method is automatically set to 'bayesian' otherwise it
    %                     is set to 'mle'. You can also specify 'covshrink'.
    %
    % OUTPUT:
    %
    % obj        -         the fitted object. 
    
        [data,suffStat,method] = process_options(varargin,...
            'data',[],'suffStat',[],'method','default');                        %#ok
        
        if(strcmp(method,'default'))
           
            if(isa(obj.mu,'double') && isa(obj.Sigma,'double'))
               method = 'mle'; 
            else
               method = 'bayesian'; 
            end
        end
        
        switch(lower(method))
            case 'bayesian'
                obj = fitBayesian(obj,varargin{:});
            otherwise
                obj = fitMAP(obj,varargin{:});
        end
    end
   
    
    function [postmu, logevidence] = softCondition(pmu, py, A, y)
      % Bayes rule for MVNs
      Syinv = inv(py.Sigma);
      Smuinv = inv(pmu.Sigma);
      postSigma = inv(Smuinv + A'*Syinv*A);
      postmu = postSigma*(A'*Syinv*(y-py.mu) + Smuinv*pmu.mu);
      postmu = MvnDist(postmu, postSigma);
      %evidence = mvnpdf(y(:)', (A*pmu.mu + py.mu)', py.Sigma + A*pmu.Sigma*A');
      logevidence = logprob(MvnDist(A*pmu.mu + py.mu, py.Sigma + A*pmu.Sigma*A'), y(:)');
    end
    
    
    function xrange = plotRange(obj, sf)
        if nargin < 2, sf = 3; end
        %if ndimensions(obj) ~= 2, error('can only plot in 2d'); end
        mu = mean(obj); C = cov(obj);
        s1 = sqrt(C(1,1));
        x1min = mu(1)-sf*s1;   x1max = mu(1)+sf*s1;
        if ndimensions(obj)==2
            s2 = sqrt(C(2,2));
            x2min = mu(2)-sf*s2; x2max = mu(2)+sf*s2;
            xrange = [x1min x1max x2min x2max];
        else
            xrange = [x1min x1max];
        end
    end
    
  end % methods

  %% Demos
  methods(Static = true)
    function suffStat = mkSuffStat(X)
      % SS.n
      % SS.xbar = 1/n sum_i X(i,:)'
      % SS.XX(j,k) = 1/n sum_i XC(i,j) XC(i,k)
      n = size(X,1);
      suffStat.n = n;
      %suffStat.X = sum(X,1)'; % column vector
      suffStat.xbar = sum(X,1)'/n; % column vector
      Xc = (X-repmat(suffStat.xbar',n,1));
      suffStat.XX = (Xc'*Xc)/n;
    end
 
       
    function plot2dMarginalFigure()
      plotGauss2dMargCond;
    end

  end


  %% Private methods
  methods(Access = 'protected')
    function checkParamsAreConst(obj)
      p = isa(obj.mu, 'double') && isa(obj.Sigma, 'double');
      if ~p
        error('params must be constant')
      end
    end
    
     function obj = fitMAP(obj, varargin)
      % Point estimate of parameters
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i,:) = case i
      % suffStat -
      % method - one of {map, mle, covshrink}
      %
      % For covshrink: we use the Ledoit-Wolf formula to estimate srhinkage amount
      %  See  J. Schaefer and K. Strimmer.  2005.  A shrinkage approach to
      %   large-scale covariance matrix estimation and implications
      %   for functional genomics. Statist. Appl. Genet. Mol. Biol. 4:32.

      [X, SS, method] = process_options(...
        varargin, 'data', [], 'suffStat', [], 'method', 'mle');
      hasMissingData =  any(isnan(X(:)));
      assert(~hasMissingData)
      if isempty(SS), SS = MvnDist.mkSuffStat(X); end
      switch method
        case 'mle'
          obj.mu = SS.xbar;
          obj.Sigma = SS.XX;
        case 'covshrink',
          obj.mu =  mean(X);
          obj.Sigma =  covshrink(X);
        otherwise
          error(['bad method ' method])
      end
    end

    function obj = fitBayesian(obj, varargin)
      % Computer posterior over params
       % m = fitBayesian(model, 'name1', val1, 'name2', val2, ...)
       % Arguments are
       % data - data(i,:) = case i
       % suffStat - 
       %
      % if m.mu is of type mvnInvWishDist, and there is no missing
      % data, we compute the posterior exactly. Otherwise we call
      % m = infer(m.paramInfEng, m, data) to deal with it.
      %
       [X, SS] = process_options(...
         varargin, 'data', [], 'suffStat', []);
       hasMissingData =  any(isnan(X(:)));
       if hasMissingData
         obj = infer(obj.paramInfEng, obj, X);
         return;
       end
       if isempty(SS), SS = MvnDist.mkSuffStat(X); end
       if SS.n == 0, return; end
       done = false;
       switch class(obj.mu)
         case 'MvnDist'
           if ~isa(obj.Sigma, 'double'), error('Sigma must be a constant'); end
             %obj.mu = updateMean(obj.mu, SS, obj.Sigma);
             mu = obj.mu;
             S0 = mu.Sigma; S0inv = inv(S0);
             mu0 = mu.mu;
             S = obj.Sigma; Sinv = inv(S);
             n = SS.n;
             Sn = inv(inv(S0) + n*Sinv);
             obj.mu = MvnDist(Sn*(n*Sinv*SS.xbar + S0inv*mu0), Sn);
             done = true;
         case 'MvnInvWishartDist'
           %obj.mu = updateMeanCov(obj.mu, SS);
           mu = obj.mu;
           k0 = mu.k; m0 = mu.mu; T0 = mu.Sigma; v0 = mu.dof;
           n = SS.n;
           kn = k0 + n;
           vn = v0 + n;
           Tn = T0 + n*SS.XX + (k0*n)/(k0+n)*(SS.xbar-m0)*(SS.xbar-m0)';
           mn = (k0*m0 + n*SS.xbar)/kn;
           obj.mu = MvnInvWishartDist('mu',mn, 'Sigma', Tn, 'dof', vn, 'k', kn);
           done = true;
         case 'double'
           if ~isa(obj.Sigma, 'InvWishartDist'), error('Sigma must be InvWishart'); end
           %obj.Sigma = updateSigma(obj.Sigma, obj.mu, SS);
           mu = obj.mu; Sigma = obj.Sigma;
           n = SS.n;
           T0 = Sigma.Sigma;
           v0 = Sigma.dof;
           vn = v0 + n;
           Tn = T0 + n*SS.XX +  n*(SS.xbar-mu)*(SS.xbar-mu)';
           obj.Sigma = InvWishartDist(vn, Tn);
           done = true;
       end
       if ~done
         obj = infer(obj.paramInfEng, obj, X);
       end
    end
    
 
    
  end


end