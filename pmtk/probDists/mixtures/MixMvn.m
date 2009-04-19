classdef MixMvn < MixModel
% Mixture of Multivariate Normal Distributions    
    
methods

  function model = MixMvn(varargin)
    % model = MixMvn(nmixtures, nstates, support, transfomer)
    % Create a model with default priors for MAP estimation
    if nargin == 0; return; end
    [nmixtures, ndims, model.fitEng, model.transformer] = ...
      processArgs(varargin,...
      'nmixtures'    ,[] ,...
      'ndims',      0,...
      'fitEng',       EmMixMvnEng(), ...
      'transformer'  ,[]);
    K = nmixtures;
    %T = normalize(rand(K,1));
    alpha = 2; % MAP estimate is counts + alpha - 1
    mixingDistrib = DiscreteDist('nstates', K, 'prior','dirichlet', 'priorStrength', alpha);
    dist = MvnDist('ndims', ndims, 'prior','niw');
    distributions = copy(dist,K,1);
    model.mixingDistrib = mixingDistrib;
    model.distributions = distributions;
    model.nmix = numel(model.distributions);
  end
    
end % methods

end


