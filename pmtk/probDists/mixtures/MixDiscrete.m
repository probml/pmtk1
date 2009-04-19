classdef MixDiscrete < MixModel
  % A mixture of products of Discrete Distributions, (i.e. Naive Bayes structure,
  % but with hidden class labels). Each product of Discrete distributions is
  % represented as a single vectorized DiscreteDist object. This class can be used
  % to create a mixture of Bernoullis, simply specify nstates=2.

  methods

    function model = MixDiscrete(varargin)
      % model = MixDiscrete(nmixtures, nstates, support, transfomer)
      % Create a model with default priors for MAP estimation
      if nargin == 0; return; end
      [nmixtures,  nstates, support, model.fitEng, model.transformer]...
        = processArgs(varargin,...
        'nmixtures'    ,[] ,...
        'nstates',      [],...
        'support',      [], ...
        'fitEng',       EmMixModelEng(), ...
        'transformer'  ,[]);
      K = nmixtures;
      %T = normalize(rand(K,1));
      alpha = 2; % MAP estimate is counts + alpha - 1
      mixingDistrib = DiscreteDist('nstates', K, 'prior','dirichlet', 'priorStrength', alpha);
      % we need to know what values the features can take on
      % so we can define a distribution of the right size
      if isempty(support)
        if ~isempty(nstates)
          support = 1:nstates;
        end
      end
      if isempty(support), error('must specify support or nstates'); end
      dist = DiscreteDist('support', support, 'prior','dirichlet', 'priorStrength', alpha);
      distributions = copy(dist,K,1);
      model.mixingDistrib = mixingDistrib;
      model.distributions = distributions;
      model.nmix = numel(model.distributions);
    end
    
   


  end %  methods
end