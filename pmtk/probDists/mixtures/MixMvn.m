classdef MixMvn < MixModel
% Mixture of Multivariate Normal Distributions    
    
methods

  function model = MixMvn(varargin)
    % model = MixMvn(nmixtures, ndims, fitEng, transformer)
    % Create a model with default priors for MAP estimation
    if nargin == 0; return; end
    [nmixtures, ndims, distcov, distprior, model.fitEng, model.transformer, model.verbose] = ...
      processArgs(varargin,...
      '-nmixtures'    ,[] ,...
      '-ndims',      0,...
      '-covtype',    'full', ...
      '-prior',      'niw', ...
      '-fitEng',       EmMixMvnEng(), ...
      '-transformer'  ,[], ...
      '-verbose',      false);
    K = nmixtures;
    T = normalize(ones(K,1));
    alpha = 2; % MAP estimate is counts + alpha - 1
    mixingDistrib = DiscreteDist('-T', T, '-prior','dirichlet', '-priorStrength', alpha);
    dist = MvnDist('-ndims', ndims, '-prior', distprior, '-covtype', distcov);
    distributions = copy(dist,K,1);
    model.mixingDistrib = mixingDistrib;
    model.distributions = distributions;
    model.nmix = numel(model.distributions);
  end

  function model = EMfit(model, X, varargin)
    [distributions, mix] = EMforGMM(model.distributions, model.mixingDistrib, X, varargin{:});
    model.distributions = distributions;
    model.mixingDistrib.T = mix;
  end
    
end % methods

end


