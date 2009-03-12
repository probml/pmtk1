classdef GaussInfEng < InfEng
% Exact inference in multivariate Gaussians
% Model must support the following method
% [mu,Sigma,domain] = convertToMvn(model);
 
properties
  mu; Sigma; domain;
end



  %% main methods
  methods
    function eng = GaussInfEng()
      eng.mu = []; eng.Sigma = []; eng.domain = [];
    end
     
    function [eng, logZ, other] = condition(eng, model, visVars, visValues)
      %if ~isempty(model.discreteNodes)
      %  error('GaussInfEng requires all  nodes to be Gaussian')
      %end
      [mu, Sigma, domain] = convertToMvn(model);
      V = lookupIndices(visVars, domain);
      hidVars = setdiffPMTK(domain, visVars);
      [muHgivenV, SigmaHgivenV] = gaussianConditioning(...
        mu, Sigma, V, visValues);
      eng.mu = muHgivenV; eng.Sigma = SigmaHgivenV; eng.domain = hidVars;
      logZ = []; other = [];
    end
    
    function [postQuery,eng] = marginal(eng, queryVars)
       mu = eng.mu; Sigma = eng.Sigma; domain = eng.domain;
       Q = lookupIndices(queryVars, domain);
       postQuery = MvnDist(mu(Q), Sigma(Q,Q), 'domain', queryVars); %#ok
    end
  
    function samples = sample(eng, n)
      % Samples(i,:) is i'th sample of the *hidden* nodes 
      if(nargin < 2), n = 1; end;
      mu = eng.mu; Sigma = eng.Sigma; 
      A = chol(Sigma,'lower'); % could be cached
      Z = randn(length(mu),n);
      samples = bsxfun(@plus,mu(:), A*Z)'; %#ok
    end
   

    
  end % methods

end