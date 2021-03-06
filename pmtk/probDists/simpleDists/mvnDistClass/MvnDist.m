classdef MvnDist < ProbDist
  % multivariate normal p(X|mu,Sigma)

  properties
    mu; Sigma;
    prior;
    fitMethod;
    fitArgs;
    domain;
    infEng;
    covtype;
    ndims;
    %discreteNodes;
    %ctsNodes;
  end

  %% main methods
  methods


    function m = MvnDist(varargin)
      % MvnDist(mu, Sigma, ndims, domain, prior, fitMethod, fitArgs, covtype,
      % infEng)
      if nargin == 0; return ; end
      [m.mu, m.Sigma, m.ndims, m.domain, m.prior, m.fitMethod, m.fitArgs,...
        m.covtype, m.infEng] = processArgs(varargin, ...
        '-mu', [], '-Sigma', [], '-ndims', 0, ...
        '-domain', [], '-prior', 'none', '-fitMethod', 'mle', ...
        '-fitArgs', {}, '-covtype', 'full', '-infEng', GaussInfEng());
      if m.ndims==0, m.ndims = length(m.mu); end;
      %if m.ndims==0, error('must specify ndims and/or mu'); end
      m.domain = 1:m.ndims;
    end

     function np = dof(model)
       d = model.ndims;
       switch model.covtype
         case 'full'
           np  = d * (d-1) / 2 + d;
         case 'diag'
           np = d + d;
         case 'spherical'
           np = 1 + d;
       end
     end
     
    function mu = mean(model)
      mu = model.mu;
    end

    function mu = mode(model)
      mu = model.mu;
    end

    function C = cov(model)
      C = model.Sigma;
    end

    function C = var(model)
      C = diag(model.Sigma);
    end

    function [mu,Sigma,domain] = convertToMvn(m)
      % This is required by  the GaussInfEng object
      mu = m.mu; Sigma = m.Sigma;
      domain = 1:length(m.mu);
    end

    function [samples, other] = sample(model, n, visVars, visVals)
      % Samples(i,:) is i'th sample
      if(nargin < 2), n = 1; end;
      if nargin < 3, visVars = []; visVals = []; end
      [eng, logZ, other] = condition(model.infEng, model, visVars, visVals);
      [samples] = sample(eng, n);
    end

   
    function [postQuery, logZ, other] = marginal(model, queryVars, visVars, visVals)
      if nargin < 3, visVars = []; visVals = []; end
      [eng, logZ, other] = condition(model.infEng, model, visVars, visVals);
      if ~iscell(queryVars)
        [postQuery] = marginal(eng, queryVars);
      else
        for q=1:length(queryVars)
          postQuery{q} = marginal(eng, queryVars{q}); %#ok
        end
      end
    end

    
    function logZ = lognormconst(model)
      mu = model.mu; Sigma = model.Sigma; d = length(mu);
      logZ = (d/2)*log(2*pi) + 0.5*logdet(Sigma); % could be pre-computed
    end

    
    function L = logprob(model,X, normalize)
      % L(i) = log p(X(i,:) | params)
      if nargin < 3, normalize = true; end
      mu = model.mu; Sigma = model.Sigma;
      d = length(mu);
      XC = bsxfun(@minus,X,rowvec(mu));
      L = -0.5*sum((XC*inv(Sigma)).*XC,2);
      if normalize
        logZ = (d/2)*log(2*pi) + 0.5*logdet(Sigma);
        L = L - logZ;
      end
      if false % debugging
        SS = MvnDist.mkSuffStat(X);
        LL = logprobSS(model, SS);
        assert(approxeq(LL, sum(L)));
      end
    end

    
    function L = logprior(model)
      if strcmp(model.prior, 'none') || isa(model.prior, 'NoPrior')
        L = 0;
      else 
        L = logprob(model.prior, model.mu, model.Sigma);
      end
    end


    function fc = makeFullConditionals(obj, visVars, visVals)
      % needed by GibbsInfEng
      d = length(obj.mu);
      if nargin < 2
        % Sample from the unconditional distribution
        visVars = []; visVals = [];
      end
      V = visVars; H = setdiffPMTK(1:d, V);
      x = zeros(1,d); x(V) = visVals;
      fc = cell(length(H),1);
      for i=1:length(H)
        fc{i} = @(xh) fullCond(obj, xh, i, H, x);
      end
    end

    function p = fullCond(obj, xh, i, H, x)
      % called by makeFullConditionals
      assert(length(xh)==length(H))
      x(H) = xh; % insert sampled hidden values into hidden slot
      x(i) = []; % remove value for i'th node, which will be sampled
      d = length(obj.mu);
      dom = 1:d; dom(i) = []; % specify that all nodes are observed except i
      [muAgivenB, SigmaAgivenB] = gaussianConditioning(obj.mu, obj.Sigma, dom, x);
      %xi = normrnd(muAgivenB, sqrt(SigmaAgivenB));
      p = GaussDist(muAgivenB, SigmaAgivenB);
    end

    function xinit = mcmcInitSample(model, visVars, visVals)
       % needed by GibbsInfEng, MhInfEng
      if nargin < 2
        xinit = mvnrnd(model.mu, model.Sigma);
        return;
      end
      % Ideally we would draw an initial sample conditional on the
      % observed data.
      % Instead we sample the hidden nodes from their prior
      domain = model.domain;
      hidVars = setdiffPMTK(domain, visVars);
      V = lookupIndices(visVars, domain);
      H = lookupIndices(hidVars, domain);
      xinit = mvnrnd(model.mu(H), model.Sigma(H,H));
    end

    function obj = mkRndParams(obj, d)
      if nargin < 2, d = ndimensions(obj); end
%{ Why is this here?  Is it ever called with data?
      if(~isscalar(d) || d~=round(d))
        % d is data n*D
        fprintf('mkRndParams called with data\n');

        perm = randperm(size(d,1));
        obj.mu = d(perm(1),:);
        obj.Sigma = 0.05*cov(d);
        obj.domain = 1:size(d,2);
      else
        obj.mu = randn(d,1);
        obj.Sigma = randpd(d);
        obj.domain = 1:d;
        obj.ndims = d;
      end
%}

      obj = MvnDist('-mu', randn(d,1), ...
        '-Sigma',     randpd(d), ...
        '-prior',     'niw', ...
        '-fitMethod', 'mle', ...
        '-fitArgs',   {}, ...
        '-covtype',   'full', ...
        '-infEng',    GaussInfEng());
      obj.ndims = d;
      obj.prior = 'niw';
      obj.fitMethod = 'mle';
      obj.fitArgs = {};
      obj.covtype = 'full';
      obj.infEng = GaussInfEng();
    end

    function d = ndimensions(m)
      %d= length(m.mu);
      d = m.ndims;
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
      % 'prior'    -        This can be a string chosen from
      %       {'none', 'covshrink', 'niw', 'nig'}
      %                     or an object chosen from
      %       { NoPrior, MvnInvWishartDist, MvnInvGammaDist }
      %  In the context of 'none' or NoPrior we compute the MLE, otherwise a MAP estimate.
      %
      % 'covtype'  -        Restrictions on the covariance: 'full' | 'diag' |
      %                     'spherical'

      [X,SS,prior,covtype, fitArgs,fitMethod] = process_options(varargin,...
        'data'              ,[]         ,...
        'suffStat'          ,[]         ,...
        'prior'             ,obj.prior,...
        'covtype'           ,obj.covtype, ...
        'fitArgs'           , obj.fitArgs, ...
        'fitMethod'         , obj.fitMethod);
      if any(isnan(X))
        obj = fitMvnEcm(obj, X, prior, fitArgs{:}); return;
      end
      %obj = MvnDist();
      obj.covtype = covtype;
      if isempty(X) && isempty(SS)
        error('must call fit(model,''data'',data)')
      end
      if isempty(SS), SS = MvnDist().mkSuffStat(X); end
      if isa(prior, 'char'),  prior = mkPrior(obj, '-suffStat', SS, '-prior', obj.prior, '-covtype', obj.covtype); end      
      if isempty(prior), prior = NoPrior; end
      obj.prior = prior; % replace string with object so logprior(model) works
  
      switch class(prior)
        case 'NoPrior',
           obj.mu = SS.xbar;
           obj.Sigma = SS.XX;
        otherwise
          tmp = MvnConjugate(obj,'prior', prior);
          tmp = fit(tmp, 'suffStat', SS, 'covtype', covtype); % We need to pass in covtype in the case that the prior is an MvnInvGammaDist (spherical or diagonal?)
          post = tmp.muSigmaDist; % paramDist(m); % NIW
          m = mode(post, covtype);
          obj.mu = m.mu;
          obj.Sigma = m.Sigma;
      end
    end % fit
    
    function [postmu, logevidence] = softCondition(pmu, py, A, y)
      % Bayes rule for MVNs
      Syinv = inv(py.Sigma);
      Smuinv = inv(pmu.Sigma);
      postSigma = inv(Smuinv + A'*Syinv*A);
      postmu = postSigma*(A'*Syinv*(y-py.mu) + Smuinv*pmu.mu);
      postmu = MvnDist(postmu, postSigma);
      %evidence = mvnpdf(y(:)', (A*pmu.mu + py.mu)', py.Sigma + A*pmu.Sigma*A');
      if nargout > 1
        logevidence = logprob(MvnDist(A*pmu.mu + py.mu, py.Sigma + A*pmu.Sigma*A'), y(:)');
      end
    end


    function xrange = plotRange(obj, sf)
      if nargin < 2, sf = 3; end
      %if ndimensions(obj) ~= 2, error('can only plot in 2d'); end
      mu = mean(obj); C = cov(obj);
      s1 = sqrt(C(1,1));
      x1min = mu(1)-sf*s1;   x1max = mu(1)+sf*s1;
      switch length(mu) % ndimensions(obj)
        case 1,  xrange = [x1min x1max];
        case 2,
          s2 = sqrt(C(2,2));
          x2min = mu(2)-sf*s2; x2max = mu(2)+sf*s2;
          xrange = [x1min x1max x2min x2max];
        otherwise
          d = ndimensions(obj);
          xrange = zeros(d,d,4);
          stdev = sqrt(diag(C));
          for i=1:d
            for j=1:i
              xrange(i,j,:) = [mu(i) - sf*stdev(i), mu(i) + sf*stdev(i), mu(j) - sf*stdev(j), mu(j) + sf*stdev(j)];
              xrange(j,i,:) = [mu(j) - sf*stdev(j), mu(j) + sf*stdev(j), mu(i) - sf*stdev(i), mu(i) + sf*stdev(i)];
            end
          end
      end
    end


    function [Xc, V] = impute(model, X)
      % Fill in NaN entries of X using posterior mode on each row
      % Xc(i,j) = E[X(i,j) | D]
      % V(i,j) = Variance
      [n,d] = size(X);
      Xc = X;
      V = zeros(n,d);
      for i=1:n
        hidNodes = find(isnan(X(i,:)));
        if isempty(hidNodes), continue, end;
        visNodes = find(~isnan(X(i,:)));
        visValues = X(i,visNodes);
        postH = marginal(model, hidNodes, visNodes, visValues);
        Xc(i,hidNodes) = rowvec(mode(postH));
        V(i,hidNodes) = rowvec(var(postH));
      end
    end

     function model = initPrior(model,data)
       model.prior = mkPrior(model, '-data', data);
     end

    function priorObj = priorobject(obj, priorStr)
      table = { 'none'  , NoPrior(); ...
                'niw'   , MvnInvWishartDist(); ...
                'nig'   , MvnInvGammaDist()};
      table = cell(table);
      loc = find(strcmpi(priorStr, table(:,1)));
      priorObj = table{loc,2};
    end
     

    function priorDist = mkPrior(obj,varargin)
      [data, suff, prior, covtype] = processArgs(varargin, '-data', [], '-suffStat', [], '-prior', obj.prior, '-covtype', obj.covtype);
      %if(isempty(data) && isempty(suff)), return; end;
      % If the user provides sufficient stats, then use these to make the prior
      if(~isempty(suff))
        n = suff.n;
        XX = suff.XX;
        xbar = suff.xbar;
      else
        n = rows(data);
        XX = nancov(data) + 0.1*mean(diag(nancov(data)));
        xbar = nanmean(data);
      end
      d = numel(xbar);
      if(isa(prior, 'char') || isa(prior, 'double')), prior = priorobject(obj, prior); end;
      kappa0 = 0.01; m0 = xbar;
      switch class(prior)
        % It will be double if for some reason it was not initialized, ie, obj.prior = [];
        case {'NoPrior', 'double'}
          priorDist = NoPrior();
        case 'MvnInvWishartDist'
          nu0 = d + 1; T0 = diag(diag(XX)) + 0.01*eye(d);
          priorDist = MvnInvWishartDist('mu', m0, 'Sigma', T0, 'dof', nu0, 'k', kappa0);
        case 'MvnInvGammaDist'
          switch lower(covtype)
            case 'diagonal'
              nu0 = 2*ones(1,d); b0 = rowvec(diag(XX)) + 0.01*ones(1,d);
              priorDist = MvnInvGammaDist('mu', m0, 'Sigma', kappa0, 'a', nu0, 'b', b0);
            case 'spherical'
              nu0 = 2; b0 = mean(diag(XX)) + 0.01;
              priorDist = MvnInvGammaDist('mu', m0, 'Sigma', kappa0, 'a', nu0, 'b', b0);
            otherwise
              error('MvnDist:mkPrior:invalidCombo','Error, invalid combination of prior and covtype');
          end
      end
    end


    function priorlik = MvnConjugate(obj,varargin)
      [prior] = process_options(varargin, 'prior', MvnInvWishartDist());
      switch class(prior)
        case 'MvnInvWishartDist'
          priorlik = Mvn_MvnInvWishartDist(prior);
        case 'MvnInvGammaDist'
          priorlik = Mvn_MvnInvGammaDist(prior);
      end
    end
   
  end % methods

  methods(Static = true)

    function suffStat = mkSuffStat(X,weights)
      % SS.n
      % SS.xbar = 1/n sum_i X(i,:)'
      % SS.XX(j,k) = 1/n sum_i XC(i,j) XC(i,k) - centered around xbar
      % SS.XX2(j,k) = 1/n sum_i X(i,j) X(i,k)  - not mean centered
      if(nargin > 1) % weighted sufficient statistics, e.g. for EM
        suffStat.n = sum(weights,1);
        suffStat.xbar = sum(bsxfun(@times,X,weights))'/suffStat.n;  % bishop eq 13.20
        suffStat.XX2 = bsxfun(@times,X,weights)'*X/suffStat.n;
        X = bsxfun(@minus,X,suffStat.xbar');
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
        n = size(X,1);;
        suffStat.n = n;
        suffStat.xbar = sum(X,1)'/n; % column vector
        suffStat.XX2 = (X'*X)/n;
        X = bsxfun(@minus,X,suffStat.xbar');
        suffStat.XX = (X'*X)/n;
      end
    end

    function [] = plotData(X)
      [n,d] = size(X);
      figure;
      for row=1:d
        for col=row:d
        subplot2(d,d,row,col);
          if(row == col)
          	hist(X(:,row)); title(sprintf('Histogram for dimension %d', row));
          else
          	scatter(X(:,row),X(:,col)); xlabel(sprintf('Dimension %d',row')); ylabel(sprintf('Dimension %d',row'));
          end
        end
      end
    end


  end



end
