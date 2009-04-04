classdef MixMvnVBEM < ProbDist
 
  % Variational Bayes for a mixture of multivariate normals
  
  properties
    nrestarts;
    % we do not store distributions or mixingDistrib.
    % Instead, we store posterior over their parameters, represented
    % as hyper params - see Bishop ch 10
    
    alpha;
    mu;
    beta;
    W;
    v;

  end

  methods

    function model = MixMvnVBEM(varargin)
    end

    function model = fit(model, data)
      error('not yet implemented')
      model = gmmVBEM;
    end
    
    function [ph, LL] = conditional(model,data)
      % ph(i,k) = (1/S) sum_s p(H=k | data(i,:),params(s)), a DiscreteDist
      % This is the posterior responsibility of component k for data i
      % LL(i) = log p(data(i,:) | params)  is the log normalization const
    end

     function logp = logprob(model,data)
     end
    

  end % methods

end