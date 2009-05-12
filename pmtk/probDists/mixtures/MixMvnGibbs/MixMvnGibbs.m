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

    function model = mean(model)
      sampleDists = model.samples;
      % Performs full Bayes Model averaging by averaging over each sampled mu, Sigma, and mixing weights
      mu = samplesDists.mu.samples;
      Sigmatmp = sampleDists.Sigma.samples;
      mix = sampleDists.mixingWeights.samples;
      [N, d, K] = size(mu);
      Sigma = recoversigma(model);
      % perform model averaging
      muAvg = mean(dists.muDist);
      SigmaAvg = mean(Sigma,3);
      mixAvg = mean(dists.mixDist);
      for k=1:K
        model.distributions{k}.mu = colvec(muAvg(:,k));
        model.distributions{k}.Sigma = SigmaAvg(:,:,:,k);
      end
      model.mixingDistrib.T = colvec(mixAvg);
    end

    function [model, latent] = gibbssample(model,X,varargin)
      [method, fixlatent, Nsamples, Nburnin, thin, verbose] = processArgs(varargin, ...
        '-method', 'collapsed', ...
        '-fixlatent', 'false', ...
        '-Nsamples', 1000, ...
        '-Nburnin', 500, ...
        '-thin', 1, ...
        '-verbose', true);
      switch lower(method)
        case 'full'
          [muS, sigmaS, mixS, latent] = latentGibbsSampleMvnMix(model.distributions, model.mixingDistrib, X, Nsamples, Nburnin, thin, verbose);
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
  
  end % methods

  methods(Access = 'protected')

    function [Sigma] = recoversigma(model)
      % The current implementation of SampleBasedDist does not allow us to store 
      % samples of matrices, and thus stores Sigma as a row vector.
      % This method allows us to recover the original Sigma
      [N,dsq, K] = size(model.samples.Sigma);
      d = sqrt(dsq); % since dsq i a d*d matrix as a row vector
      Sigma = zeros(d,d,N,K);
      for s=1:N
        for k=1:K
          Sigma(:,:,s,k) = reshape(Sigmatmp(s,:,k),d,d);
        end
      end
    end

  end % protected methods
  
end

