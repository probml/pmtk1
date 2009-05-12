classdef MixMvnVBEM < ProbDist
 
  % Variational Bayes for a mixture of multivariate normals
  
  properties
    nrestarts;
    % we do not store distributions or mixingDistrib.
    % Instead, we store posterior over their parameters, represented
    % as hyper params - see Bishop ch 10
    % The parameter defining the Dirichlet for the mixingDistrib
    alpha;
    % The following are the parameters for a MvnInvWisharDist()
    mu;
    Sigma;
    dof;
    k;

  end

  methods

    function model = MixMvnVBEM(varargin)
    [model.alpha, model.mu, model.k, model.Sigma, model.dof, model.nrestarts] = processArgs(varargin, ...
      '-alpha', [], ...
      '-mu', [], ...
      '-k', [], ...
      '-Sigma', [], ...
      '-dof', [], ...
      '-nrestarts', 1);
    end

    function model = fit(model, data)
      %error('not yet implemented')
      for r=1:model.nrestarts
        [param{r}, L(r)] = VBforMixMvn(model.alpha, model.mu, model.k, model.Sigma, model.dof, data);
      end
      %model = gmmVBEM;
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