classdef MixMvnGibbs < MixMvn
 
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

    function model = setParamsAlt(model, mixingWeights, distributions)
      model.mixingWeights = colvec(mixingWeights);
      model.distributions = distributions;
    end
    
    function pred = predict(model,data) % old name
      pred = inferLatent(model,data);
    end

    function model = mean(model, dists)
      % Performs full Bayes Model averaging by averaging over each sampled mu, Sigma, and mixing weights
      mu = dists.muDist.samples;
      Sigmatmp = dists.SigmaDist.samples;
      mix = dists.mixDist.samples;
      %latent = dists.latentDist.samples;

      [N, d, K] = size(mu);
      % Need to post-process Sigma
      Sigma = zeros(d,d,N,K);
      for s=1:N
        for k=1:K
          Sigma(:,:,s,k) = reshape(Sigmatmp(s,:,k)',d,d)'*reshape(Sigmatmp(s,:,k)',d,d);
        end
      end
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

    function [dists] = latentGibbsSample(model,X,varargin)
      dists = latentGibbsSampleMvnMix(model.distributions, model.mixingDistrib, X, varargin{:});
    end
    
    function dists = collapsedGibbs(model,X,varargin)
      dists = collapsedGibbsSampleMvnMix(model.distributions, model.mixingDistrib, X, varargin{:});
    end

    function muPredDist = preditMu(model, muDist)
      mus = muDist.samples;
      K = ndistrib(muDist);
      muPredDist = copy(MvnDist(), K, 1);
      
    end
    
       
    
  end % methods
  
end

