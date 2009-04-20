classdef MixMvnGibbs < MixMvn
 
  % Gibbs sampling for a mixture of multivariate normals
  
  methods
     function model = MixMvnGibbs(varargin)
       if nargin == 0; return; end
      [nmixtures,mixingWeights,distributions,model.transformer,model.verbose] = process_options(varargin,...
        'nmixtures',[],'mixingWeights',[],'distributions',[],'transformer',[],'verbose',true);
      if(~isempty(distributions)),nmixtures = numel(distributions);end

      if(isempty(mixingWeights) && ~isempty(nmixtures))
        mixingWeights = DiscreteDist('-T',normalize(ones(nmixtures,1)));
      end
      model.mixingDistrib = mixingWeights;
      if(isempty(distributions)&&~isempty(model.mixingDistrib))
        distributions = copy(MvnDist(),nstates(model.mixingWeights),1);
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

    function [dists] = latentGibbsSample(model,X,varargin)
      dists = latentGibbsSampleMvnMix(model.distributions, model.mixingDistrib, X, varargin{:});
    end
    
    function dists = collapsedGibbs(model,X,varargin)
      dists = collapsedGibbsSampleMvnMix(model.distributions, model.mixingDistrib, X, varargin{:});
    end
    
       
    
  end % methods
  
end

