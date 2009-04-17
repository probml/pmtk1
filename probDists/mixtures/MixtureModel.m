classdef MixtureModel
 
  % Mixtures of any probability distributions can be created with this
  % class. We store point estimates of the parameters.

  properties
    distributions;      % a cell array storing the class conditional densities
    mixingDistrib;      % DiscreteDist (may have prior inside of it)
    transformer;        % data preprocessor
    fitEng;              % Some parameter optimizer
    nmix;
  end

  methods

    function model = MixtureModel(varargin)
      % MixtureModel(nmixtures, fitEng, distributions, mixingDistrib, transformer)
      [model.nmix, model.fitEng, model.distributions, ...
        model.mixingDistrib, model.transformer]...
        = processArgs(varargin,...
        'nmixtures'    ,[] ,...
        'fitEng',       EmEng(), ...
        'distributions',[] ,...
        'mixingDistrib',[] ,...
        'transformer'  ,[]);
      if isempty(model.nmix)
        model.nmix = length(model.distributions);
      end
      if isempty(model.nmix)
        error('must specify nmixtures or distributions')
      end
    end


    function [ph, LL] = inferLatent(model,data)
      % ph(i,k) = p(H=k | data(i,:),params) a DiscreteDist
      % This is the posterior responsibility of component k for data i
      % LL(i) = log p(data(i,:) | params)  is the log normalization constat
      logRik = calcResponsibilities(model, data);
      [Rik, LL] = normalizeLogspace(logRik);
      Rik = exp(Rik);
      ph = DiscreteDist('T',Rik');
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
    
    function mu = mean(m)
      % mu(:) = sum_k p(k) mu(:,k)
      K = nmixtures(m);
      mu = zeros(ndimensions(m),K);
      for k=1:K
        mu(:,k) = mean(m.distributions{k});
      end
      mixWeights = pmf(model.mixingDistrib);
      M = bsxfun(@times,  mu, rowvec(mixWeights));
      mu = sum(M, 2);
    end

    function C = cov(m)
      % Cov(:,:) = sum_k p(k) (Cov(:,:,k) + mu(:,k)*mu(:,k)') - mu*mu'
      K = nmixtures(m);
      d = ndimensions(m);
      mixWeights = pmf(model.mixingDistrib);
      C = zeros(d,d);
      for k=1:K
        mu = mean(m.distributions{k});
        C = C + mixWeights(k)*(cov(m.distributions{k})+ mu*mu');
      end
      mu = mean(m);
      C = C - mu*mu';
    end

    function [model, LL, niter] = fit(model, data)
      [model, LL, niter] = fit(model.fitEng, model, data);
    end
    
    %% Methods needed by EmEng
    
    function [ess, L] = Estep(model, data)
      [Rik, LL]  = inferLatent(model, data);
      N = size(data,1);
      L = sum(LL);
      %assert(approxeq(L, sum(logprob(model,data))));
      L = L + logprior(model); % must add log prior for MAP estimation
      L = L/N;
      Rik = pmf(Rik)'; % convert from distribution to table of n*K numbers
      %ess.data = data; % shouldn't have to pass this around!!
      K = length(model.distributions);
      compSS = cell(1,K);
      for k=1:K
          compSS{k} = model.distributions{k}.mkSuffStat(data,Rik(:,k));
      end
      ess.compSS = compSS;
      ess.counts = colvec(normalize(sum(Rik,1)));
    end
  
    function model = Mstep(model, ess)
      K = length(model.distributions);
      %Rik = ess.Rik;
      %data = ess.data; % yuck!
      for k=1:K
        %essK = model.distributions{k}.mkSuffStat(data,Rik(:,k));
        model.distributions{k} = fit(model.distributions{k},'suffStat',ess.compSS{k});
      end
      mixSS.counts = ess.counts;
      model.mixingDistrib = fit(model.mixingDistrib,'suffStat',mixSS);
    end
    
    function model = initializeEM(model,data,r) %#ok that ignores data and r
      % override in subclass with more intelligent initialization
      K = nmixtures(model);
      model = initPrior(model, data);
      %model = mkRndParams(model);
      % Fit k'th class conditional to k'th random portion of the data.
      % Fit mixing weights to be random.
      % (This is like initializing the assignments randomyl and then
      % doing an M step)
      n = size(data,1);
      model.mixingDistrib = mkRndParams(model.mixingDistrib);
      perm = randperm(n);
      batchSize = max(1,floor(n/K));
      for k=1:K
        start = (k-1)*batchSize+1;
        initdata = data(perm(start:start+batchSize-1),:);
        model.distributions{k} = fit(model.distributions{k},'data',initdata);
      end
    end
    
   
    function displayProgress(model,data,loglik,iter,r) %#ok that ignores model
      % override in subclass with more informative display
      t = sprintf('EM restart %d iter %d, negloglik %g\n',r,iter,-loglik);
      fprintf(t);
    end

  end % methods

end

