classdef GaussInfEng < InfEng
% Exact inference in multivariate Gaussians
  

  %% main methods
  methods
    function obj = GaussInfEng()
      % The engine maintains no internal state.
    end
    
    function samples = sample(eng, model, n)%#ok
      % Samples is size n * ndims
      if(nargin < 2), n = 1; end;
      mu = model.mu; Sigma = model.Sigma; domain = model.domain;
      A = chol(Sigma,'lower'); % could be cached
      Z = randn(length(mu),n);
      samples = bsxfun(@plus,mu(:), A*Z)';
    end
     
     function postQuery = marginal(eng, model, queryVars)
       % prob = sum_h p(Query,h)
       if ~iscell(queryVars)
         postQuery = marginalNoCell(eng, model,queryVars);
       else
         for i=1:length(queryVars)
           postQuery{i} = marginalNoCell(eng, model, queryVars{i});
         end
       end
     end
     
     function postQuery = marginalNoCell(eng, model, queryVars)%#ok
       % prob = sum_h p(Query,h)
       mu = model.mu; Sigma = model.Sigma; domain = model.domain;
       Q = lookupIndices(queryVars, domain);
       postQuery = MvnDist(mu(Q), Sigma(Q,Q), 'domain', queryVars);
     end
     
     function prob = predict(eng, model, visVars, visValues, queryVars)%#ok
       %prob =  sum_h p(query,h|visVars=visValues)
       mu = model.mu; Sigma = model.Sigma; domain = model.domain;
       V = lookupIndices(visVars, domain);
       hidVars = mysetdiff(domain, visVars);
       [muHgivenV, SigmaHgivenV] = gaussianConditioning(...
         mu, Sigma, V, visValues);
       prob = MvnDist(muHgivenV, SigmaHgivenV, 'domain', hidVars);
       if (nargin >= 4) && ~isempty(queryVars)
         prob = marginal(prob, queryVars);
       end
     end
  
    
   
  end % methods

end