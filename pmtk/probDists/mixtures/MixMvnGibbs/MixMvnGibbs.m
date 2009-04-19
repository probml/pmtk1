classdef MixMvnGibbs < MixMvn
 
  % Gibbs sampling for a mixture of multivariate normals
  
  methods
     function model = MvnMixDist(varargin)
      [nmixtures,mixingWeights,distributions,model.transformer,model.verbose,model.nrestarts] = process_options(varargin,...
        'nmixtures',[],'mixingWeights',[],'distributions',[],'transformer',[],'verbose',true,'nrestarts',model.nrestarts);
      if(~isempty(distributions)),nmixtures = numel(distributions);end

      if(isempty(mixingWeights) && ~isempty(nmixtures))
        mixingWeights = DiscreteDist('T',normalize(ones(nmixtures,1)));
      end
      model.mixingWeights = mixingWeights;
      if(isempty(distributions)&&~isempty(model.mixingWeights))
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
      dists = latentGibbsSampleMnvMix(model,X,varargin{:});
    end
    
    function [dists] = collapsedGibbs(model,X,varargin)
      dists = collapsedGibbsSampleMnvMix(model,X,varargin{:});
    end
    
       
    
  end % methods
  
end

