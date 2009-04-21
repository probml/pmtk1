classdef MixModel < ProbDist
 
  % Mixtures of any probability distributions can be created with this
  % class. We store point estimates of the parameters.

  properties
    distributions;      % a cell array storing the class conditional densities
    mixingDistrib;      % DiscreteDist (may have prior inside of it)
    transformer;        % data preprocessor
    fitEng;              % Some parameter optimizer
    nmix;
    verbose;
  end

  methods

    function model = MixModel(varargin)
      % MixtureModel(nmixtures, fitEng, distributions, mixingDistrib, transformer)
      [model.nmix, model.fitEng, model.distributions, ...
        model.mixingDistrib, model.transformer, model.verbose]...
        = processArgs(varargin,...
        '-nmixtures'    ,[] ,...
        '-fitEng',       EmMixModelEng(), ...
        '-distributions',[] ,...
        '-mixingDistrib',[] ,...
        '-transformer'  ,[],...
        '-verbose'       , false);
      if isempty(model.nmix)
        model.nmix = length(model.distributions);
      end
      if isempty(model.nmix)
        error('must specify nmixtures or distributions')
      end
      if isempty(model.mixingDistrib)
        model.mixingDistrib = DiscreteDist('-T',normalize(ones(model.nmix,1)));
      end
    end


     function logRik = calcResponsibilities(model,data)
      % returns *unnormalized* log responsibilities
      % logRik(i,k) = log p(data(i,:), hi=k | params) 
      if(~isempty(model.transformer))
        data = test(model.transformer,data);
      end
      n = size(data,1); nmixtures = numel(model.distributions);
      logRik = zeros(n,nmixtures);
      mixWeights = pmf(model.mixingDistrib);
      for k=1:nmixtures
        logRik(:,k) = log(mixWeights(k)+eps) + logprob(model.distributions{k},data);
      end
     end
    
    function [ph, LL] = inferLatent(model,data)
      % ph(i,k) = p(H=k | data(i,:),params) a DiscreteDist
      % This is the posterior responsibility of component k for data i
      % LL(i) = log p(data(i,:) | params)  is the log normalization constat
      logRik = calcResponsibilities(model, data);
      [Rik, LL] = normalizeLogspace(logRik);
      Rik = exp(Rik);
assert(approxeq(Rik, normalize(Rik,2)))
      ph = DiscreteDist('-T',Rik');
    end
    

    function logp = logprob(model,data)
      % logp(i) = log p(data(i,:) | params)
      %  = log sum_k p(data(i,:), h=k | params) 
      logp = logsumexp(calcResponsibilities(model, data),2);
    end

    function L = logprior(model)
      L = logprior(model.mixingDistrib);
      for k=1:nmixtures(model)
        L = L + sum(logprior(model.distributions{k}));
      end
    end
    

    function model = mkRndParams(model)
      % create K random class-conditional densities of size d
      % and a random mixing distribution
      for i=1:nmixtures(model)
        model.distributions{i} = mkRndParams(model.distributions{i});
      end
      model.mixingDistrib = mkRndParams(model.mixingDistrib);
    end

    function model = initPrior(model, data)
      % Convert prior strings into distributions, possibly using the data
      K = nmixtures(model);
      for i=1:K
        model.distributions{i} = initPrior(model.distributions{i}, data);
      end
      model.mixingDistrib = initPrior(model.mixingDistrib, data);
    end
    

    function [Y, H] = sample(model,nsamples)
      % Y(i,:) = i'th sample of observed nodes
      % H(i) = i'th sample of hidden node
      if nargin < 2, nsamples = 1; end
      H = sample(model.mixingDistrib, nsamples);
      d = ndimensions(model);
      Y = zeros(nsamples, d);
      for i=1:nsamples
        Y(i,:) = rowvec(sample(model.distributions{H(i)}));
      end
    end

    function d = ndimensions(model)
      if(numel(model.distributions) > 0)
        d = ndimensions(model.distributions{1});
      else
        d = 0;
      end
    end

    function K = nmixtures(model)
      %K = length(model.distributions);
      K = model.nmix;
    end
    
    %{
    function d = ndistrib(model)
      d = max(1,numel(model.distributions));
    end
    %}
    
   
    
    function mu = mean(m)
      % mu(:) = sum_k p(k) mu(:,k)
      K = nmixtures(m);
      mu = zeros(ndimensions(m),K);
      for k=1:K
        mu(:,k) = mean(m.distributions{k});
      end
      mixWeights = pmf(m.mixingDistrib);
      M = bsxfun(@times,  mu, rowvec(mixWeights));
      mu = sum(M, 2);
    end

    function C = cov(m)
      % Cov(:,:) = sum_k p(k) (Cov(:,:,k) + mu(:,k)*mu(:,k)') - mu*mu'
      K = nmixtures(m);
      d = ndimensions(m);
      mixWeights = pmf(m.mixingDistrib);
      C = zeros(d,d);
      for k=1:K
        mu = mean(m.distributions{k});
        C = C + mixWeights(k)*(cov(m.distributions{k})+ mu*mu');
      end
      mu = mean(m);
      C = C - mu*mu';
    end

    function [model, LL, niter] = fit(model, varargin)
      % fit(model, data)
      [data] = processArgs(varargin, '-data', []);
      [model, LL, niter] = fit(model.fitEng, model, data);
    end
    
    function SS = mkSuffStat(model,data,weights)
      % needed by HMM/EM
      % Compute weighted, (expected) sufficient statistics. In the case of
      % an HMM, the weights correspond to gamma = normalize(alpha.*beta,1)
      % We calculate gamma2 by combining alpha and beta messages with the
      % responsibilities - see equation 13.109 in pml24nov08
      if(nargin < 2)
        weights = ones(size(data,1));
      end
      if(~isempty(model.transformer))
        [data,model.transformer] = train(model.transformer,data);
      end
      logRik = calcResponsibilities(model,data);
      logGamma2 = bsxfun(@plus,logRik,log(weights+eps));           % combine alpha,beta,local evidence messages
      %logGamma2 = bsxfun(@minus,logGamma2,logsumexp(logGamma2,2)); % normalize while avoiding numerical underflow
      logGamma2 = normalizeLogspace(logGamma2);
      gamma2 = exp(logGamma2);
      nmixtures = numel(model.distributions);
      ess = cell(nmixtures,1);
      for k=1:nmixtures
        ess{k} = model.distributions{k}.mkSuffStat(data,gamma2(:,k));
      end
      SS.ess = ess;
      SS.weights = gamma2;
    end
    
    function p = isDiscrete(CPD) %#ok
    % used by DgmDist constructor    
      p = false;
    end
    
    function q = nstates(CPD)  
    % used by DgmDist constructor    
      q = length(pmf(CPD.mixingDistrib));
    end
    
    function Tfac = convertToTabularFactor(model, child, ctsParents, dParents, visible, data, nstates,fullDomain)
    % all of the children must be observed
        assert(isempty(ctsParents))
        assert(length(dParents)==1)
        map = @(x)canonizeLabels(x,fullDomain);
        if visible(map(child))
            T = exp(calcResponsibilities(model,data(map(child))));
            Tfac = TabularFactor(T,dParents);
        else
            % barren leaf removal
            Tfac = TabularFactor(ones(1,nstates(map(dParents))), dParents);
        end
    end

    
  end % methods

end

