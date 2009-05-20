classdef MixMvnGibbs < MixMvn

  properties
    samples;
  end
 
  % Gibbs sampling for a mixture of multivariate normals
  
  methods
     function model = MixMvnGibbs(varargin)
       if nargin == 0; return; end
      [nmixtures,mixingWeights,distributions,model.transformer,model.verbose] = process_options(varargin,...
        'nmixtures',[],'mixingWeights',[],'distributions',[],'transformer',[],'verbose',true);
      if(~isempty(distributions)),nmixtures = numel(distributions);end

      if(isempty(mixingWeights) && ~isempty(nmixtures))
        mixingWeights = DiscreteDist('-T',normalize(ones(nmixtures,1)), '-prior', DirichletDist(ones(nmixtures,1)) );
      end
      model.mixingDistrib = mixingWeights;
      if(isempty(distributions)&&~isempty(model.mixingDistrib))
        distributions = copy(MvnDist(),nstates(model.mixingDistrib),1);
      end
      model.distributions = distributions;
    end
    
    function pred = predict(model,data) % old name
      pred = inferLatent(model,data);
    end

    function x = sample(model, n)
      if(isempty(model.samples)), error('MixMvnGibbs','You must call fit first'); end;
      d = numel(model.distributions{1}.mu);
      k = sample(model.mixingDistrib,n);
      mu = zeros(d,n); Sigma = zeros(d,d,n);
      for i=1:n
        mu = colvec(sample(model.samples.mu{k(i)}, 1));
        Sigma = sample(model.samples.Sigma{k(i)}, 1);
        x(i,:) = sample(MvnDist(mu,Sigma), 1);
      end
    end

    function l = logprob(model, X)
      if(isempty(model.samples)), error('MixMvnGibbs', 'You must call fit first'); end;
      K = numel(model.distributions);
      S = size(model.samples.mu{1}.samples,2);
      [n,d] = size(X);
      l = zeros(n,K,S);
      Sigma = model.samples.Sigma;
      mu = model.samples.mu;
      mixW = model.samples.mixingWeights;
      for s=1:S
        for k=1:K
          XC = bsxfun(@minus, X, mu{k}.samples(:,s)');
          l(:,k,s) = log(mixW.samples(k,s)) - 1/2*logdet(2*pi*Sigma{k}.samples(:,:,s)) - 1/2*sum((XC*inv(Sigma{k}.samples(:,:,s))).*XC,2);
        end
      end
      l = mean(l,3);
      l = logsumexp(l,2);
    end


% We don't want this
%{
    function model = mean(model)
      sampleDists = model.samples;
      % Performs full Bayes Model averaging by averaging over each sampled mu, Sigma, and mixing weights
      mu = sampleDists.mu;
      Sigmatmp = sampleDists.Sigma;
      mix = sampleDists.mixingWeights;
      [N, d, K] = size(mu);
      Sigma = recoversigma(model);
      % perform model averaging
      SigmaAvg = mean(Sigma,3);
      mixAvg = mean(sampleDists.mixingWeights);
      for k=1:K
        model.distributions{k}.mu = colvec(mean(sampleDists.mu{k}));
        model.distributions{k}.Sigma = SigmaAvg(:,:,:,k);
      end
      model.mixingDistrib.T = colvec(mixAvg);
    end
%}

    function [model, latent] = fit(model,X,varargin)
      [method, fixlatent, Nsamples, Nburnin, thin, verbose] = processArgs(varargin, ...
        '-method', 'collapsed', ...
        '-fixlatent', 'false', ...
        '-Nsamples', 1000, ...
        '-Nburnin', 500, ...
        '-thin', 1, ...
        '-verbose', true);
      switch lower(method)
        case 'full'
          [muS, sigmaS, mixS, latent] = fullGibbsSampleMvnMix(model.distributions, model.mixingDistrib, X, Nsamples, Nburnin, thin, verbose);
        case 'collapsed'
          [muS, sigmaS, mixS, latent] = collapsedGibbsSampleMvnMix(model.distributions, model.mixingDistrib, X, Nsamples, Nburnin, thin, verbose);
      end

      if(fixlatent)
        [muS, sigmaS, mixS, latent] = processLabelSwitching(muS, sigmaS, mixS, latent, X);
      end
      model.samples.mu = muS;
      model.samples.Sigma = sigmaS;
      model.samples.mixingWeights = mixS;
    end

    function m = convertToMixMvn(model)
      m = MixMvn();
      m.distributions = model.distributions;
      m.mixingDistrib = model.mixingDistrib;
      for k=1:K
        m.distributions{k}.mu = mean(model.samples.mu{k});
        m.distributions{k}.Sigma = mean(model.samples.Sigma{k});
      end
      model.mixingDistrib.T = mean(model.samples.mixingWeights);
    end

    function [] = traceplot(model)
      if(isempty(model.samples)), error('MixMvnGibbs:traceplot:notFit', 'Must first fit Mixture to data'); end;
      mu = model.samples.mu;
      K = numel(mu);
      figure(); hold on;
      [nRows, nCols] = nsubplots(K);
      for k=1:K
        subplot(nRows, nCols, k);
        plot(mu{k}.samples);
        xlabel(sprintf('Distribution %d', k));
      end
    end
  
  end % methods
  
end

