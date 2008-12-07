classdef MvnDist < ParamDist 
% multivariate normal p(X|mu,Sigma) 
% 

%% Parameter Access
%
% obj.mu        - always returns a point estimate of mu 
% obj.Sigma     - always returns a point estimate of Sigma
%
% obj.params    - joint distribution over both mu and Sigma - if this factors, 
%                 the distribution is a ProductDist otherwise it is an 
%                 MvnInvWishartDist. 
%
% obj.params.mu - if obj.params is a ProductDist, this returns the full marginal
%                 distribution over mu. However, if obj.params is an
%                 MvnInvWishartDist, this usage is ambiguous and an error is
%                 returned. You must specify explicitly whether you want the mu
%                 property of the MvnInvWishartDist class or the marginal on mu.
%                 The former can be obtained with getParams(obj.params,'mu') and
%                 the latter with marginal(obj.params,'mu').
%
% obj.params.Sigma - same rules as obj.params.mu
  
  properties
    domain;
  end
  
  properties
     %     place holders for point estimate access such as obj.mu
     %     - actual parameters stored in params field
     mu;      
     Sigma;
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
      
      if(isa(mu,'MvnInvWishartDist'))
          m.params = mu;
      else
         if(isnumeric(mu))
             mu = ConstDist(colvec(mu));
         end
         if(isnumeric(Sigma))
             Sigma = ConstDist(Sigma);
         end
          m.params = ProductDist({mu,Sigma},{'mu','Sigma'});
      end
      m.domain = domain;
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
        if numel(obj.mu)==1
            X = X(:); % ensure column vector
        end
        [N d] = size(X);
        if length(obj.mu) ~= d
            error('X should be N x d')
        end
        if obj.Sigma==0
            L = repmat(NaN,N,1);
            return;
        end
        X = bsxfun(@minus,X,obj.mu');
        L =-0.5*sum((X*inv(obj.Sigma)).*X,2);
        if normalize
            L = L - lognormconst(obj);
        end
        if(0 && statsToolboxInstalled) % sanity check
            Lstats = log(mvnpdf(bsxfun(@plus,X,obj.mu'),obj.mu',obj.Sigma));
            assert(approxeq(L,Lstats))
        end
        
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
       if(isempty(obj.domain))
           obj.domain = 1:numel(obj.mu);
       end
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
             Xc(i,hidNodes) = rowvec(mode(postH));
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
         fc = cell(length(H),1);
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
    %                     SS = mkSuffStat(MvnDist(),data). If not specified, this
    %                     is automatically calculated.
    %
    % 'method'   -        If either obj.mu or obj.Sigma is an object, 
    %                     method is automatically set to 'bayesian' otherwise it
    %                     is set to 'mle'. You can also specify 'map' or 
    %                     'covshrink'.
    %
    % 'prior'    -        A prior over the parameters. If this is an MvnDist, it
    %                     is assumed to be a prior on mu. If it is an
    %                     InvWishartDist, it is assumed to be prior on Sigma and
    %                     if it is an MvnInvWishartDist, it is assumed to be a
    %                     prior on both. If this is not specified and
    %                     obj.params.mu and/or obj.params.Sigma are objects,
    %                     these are used as prior distributions instead. 
    %
    % 'covtype'  -        Restrictions on the covariance: 'full' | 'diag' |
    %                     'isotropic'
    %
    % OUTPUT:
    %
    % obj        -         the fitted object. 
    
        [data,suffStat,method,prior,covtype] = process_options(varargin,...
            'data'              ,[]         ,...
            'suffStat'          ,[]         ,...
            'method'            ,'default'  ,...
            'prior'             ,[]         ,...
            'covtype'           ,'full');                       
        
        if(~strcmpi(covtype,'full')),error('Restricted covtypes not yet implemnted');end
%         if(~xor(~isempty(data),~isempty(suffStat)))
%             error('Exactly one of data or suffStat must be specified');
%         end
        
        if(~isempty(prior))
           switch(class(prior))
               case 'MvnDist'
                   obj.mu = prior;
               case 'MvnInvWishartDist'
                   obj.params = prior;
               case 'InvWishartDist'
                   obj.Sigma = prior;
               otherwise
                   error('%s is not a supported prior',class(prior));
           end
        end
        
        if(strcmp(method,'default'))
            if(isa(obj.params,'ProductDist') && allConst(obj.params))
               method = 'mle';
            else
               method = 'bayesian'; 
            end 
        end
        
        switch(lower(method))
            case {'bayesian','map'}
                obj = fitBayesian(obj,varargin{:});
            otherwise
                obj = fitMLE(obj,varargin{:});
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
    
     function suffStat = mkSuffStat(obj,X,weights)
         % SS.n
         % SS.xbar = 1/n sum_i X(i,:)'
         % SS.XX(j,k) = 1/n sum_i XC(i,j) XC(i,k)
         if(nargin > 2) % weighted sufficient statistics, e.g. for EM
            
             suffStat.n = sum(weights,1);
             suffStat.xbar = sum(bsxfun(@times,X,weights))'/suffStat.n;  % bishop eq 13.20
             X = bsxfun(@minus,X,mean(X,1));
             suffStat.XX = bsxfun(@times,X,weights)'*X/suffStat.n;
             
             if(0) % sanity check
                XXtest = zeros(size(X,2));
                for i=1:size(X,1)
                    XXtest = XXtest + weights(i)*(X(i,:)'*X(i,:));       % bishop eq 13.21
                end
                XXtest = XXtest/suffStat.n;
                assert(approxeq(XXtest,suffStat.XX));
             end 
         else
             n = size(X,1);
             suffStat.n = n;
             %suffStat.X = sum(X,1)'; % column vector
             suffStat.xbar = sum(X,1)'/n; % column vector
             
             %Xc = (X-repmat(suffStat.xbar',n,1));
             %suffStat.XX = (Xc'*Xc)/n;
             X = bsxfun(@minus,X,suffStat.xbar');
             suffStat.XX = (X'*X)/n;
         end
    end
    
  end % methods

  %% Demos
  methods(Static = true)
   
 
       
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
    
     function obj = fitMLE(obj, varargin)
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

       
      [X, SS, method,prior,covtype] = process_options(...
        varargin, 'data', [], 'suffStat', [], 'method', 'mle','prior',[],'covtype','full');
      hasMissingData =  any(isnan(X(:)));
      assert(~hasMissingData)
      if isempty(SS), SS = mkSuffStat(MvnDist,X); end
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
       [X, SS,method,prior,covtype] = process_options(...
         varargin, 'data', [], 'suffStat', [],'method','prior',[],'covtype');
       hasMissingData =  any(isnan(X(:)));
       if hasMissingData
         obj = infer(obj.paramInfEng, obj, X);
         return;
       end
       if isempty(SS), SS = mkSuffStat(MvnDist(),X); end
       if SS.n == 0, return; end
       done = false;
       
       
       switch class(obj.params)
           case 'ProductDist'
               if(isa(obj.params.mu,'MvnDist'))
                   if(~isa(obj.params.Sigma,'ConstDist'))
                       error('Sigma must be a constant')
                   end
                   mu = obj.params.mu;
                   S0 = mu.Sigma; S0inv = inv(S0);
                   mu0 = mu.mu;
                   S = obj.Sigma; Sinv = inv(S);
                   n = SS.n;
                   Sn = inv(inv(S0) + n*Sinv);
                   obj.mu = MvnDist(Sn*(n*Sinv*SS.xbar + S0inv*mu0), Sn);
                   if(strcmpi(method,'map'))
                      obj.mu = obj.mu;  % looks weird but obj.mu always returns a point estimate. 
                   end
                   done = true;
               elseif(isa(obj.params.Sigma,'InvWishartDist'))
                   if(~isa(obj.params.mu,'ConstDist'))
                       error('mu must be a constant')
                   end
                   mu = obj.mu; Sigma = obj.params.Sigma;
                   n = SS.n;
                   T0 = Sigma.Sigma;
                   v0 = Sigma.dof;
                   vn = v0 + n;
                   Tn = T0 + n*SS.XX +  n*(SS.xbar-mu)*(SS.xbar-mu)';
                   obj.Sigma = InvWishartDist(vn, Tn);
                   if(strcmpi(method,'map'))
                      obj.Sigma = obj.Sigma;  % looks weird but obj.Sigma always returns a point estimate.
                   end
                   done = true;
               else
                  error('Unsupported Prior'); 
               end
               
           case 'MvnInvWishartDist'
               mu = obj.params;
               k0 = mu.k; m0 = mu.mu; T0 = mu.Sigma; v0 = mu.dof;
               n = SS.n;
               kn = k0 + n;
               vn = v0 + n;
               Tn = T0 + n*SS.XX + (k0*n)/(k0+n)*(SS.xbar-m0)*(SS.xbar-m0)';
               mn = (k0*m0 + n*SS.xbar)/kn;
               obj.params = MvnInvWishartDist('mu',mn, 'Sigma', Tn, 'dof', vn, 'k', kn);
               if(strcmpi(method,'map'))
                  obj.mu    = obj.mu;
                  obj.Sigma = obj.Sigma;
               end
               done = true;
       end
       if ~done
         obj = infer(obj.paramInfEng, obj, X);
       end
    end
  end
  
  
  %% Getters & Setters
  methods
    
      function m = get.mu(obj)
          m = mode(marginal(obj.params,'mu'));
      end
      
      function S = get.Sigma(obj)
          S = mode(marginal(obj.params,'Sigma'));
      end
      
      function obj = set.mu(obj,val)
          if(isnumeric(val))
              val = ConstDist(val);
          end
          obj.params.mu = val;
      end
      
      function obj = set.Sigma(obj,val)
          if(isnumeric(val))
              val = ConstDist(val);
          end
          obj.params.Sigma = val;
      end
  end
end