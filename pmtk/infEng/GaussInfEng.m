classdef GaussInfEng < InfEng
% Exact inference in multivariate Gaussians
  properties
    mu;
    Sigma;
    domain;
  end
  

  %% main methods
  methods
    function obj = GaussInfEng(mu, Sigma, domain)
      if nargin < 1, mu = []; end
      if nargin < 2, Sigma = []; end
      obj.mu = mu; obj.Sigma = Sigma;
      d = length(mu);
      if nargin < 3, domain = 1:d; end
      obj.domain = domain;
    end
    
    
    function samples = sample(obj,n)
      % Sample n times from this distribution: samples is of size
      % nsamples-by-ndimensions
      if(nargin < 2), n = 1; end;
      A = chol(obj.Sigma,'lower');
      Z = randn(length(obj.mu),n);
      samples = bsxfun(@plus,obj.mu(:), A*Z)';
    end
    
     
     function postQuery = marginal(eng, model, queryVars)
       % prob = sum_h p(Query,h)
       mu = model.mu; Sigma = model.Sigma; domain = model.domain;
       Q = lookupIndices(queryVars, domain);
       postQuery = MvnDist(mu(Q), Sigma(Q,Q), 'domain', queryVars); %#ok
     end
     
     %{
     function [postQuery, eng] = marginal(eng, model, queryVars)
     % prob = sum_h p(Query,h)
     mu = model.mu; Sigma = model.Sigma; domain = model.domain;
       if(isempty(obj.domain))
           obj.domain = 1:numel(obj.mu);
       end
       Q = lookupIndices(queryVars, obj.domain);
       postQuery = MvnDist(obj.mu(Q), obj.Sigma(Q,Q), 'domain', queryVars);
     end
     %}
     
     function prob = predict(obj, visVars, visValues, queryVars)
       %prob =  sum_h p(query,h|visVars=visValues)
       H = mysetdiff(obj.domain, visVars);
       [muHgivenV, SigmaHgivenV] = gaussianConditioning(...
         obj.mu, obj.Sigma, visVars, visValues);
       prob = MvnDist(muHgivenV, SigmaHgivenV, 'domain', H);
       if (nargin >= 4) && ~isempty(queryVars)
         prob = marginal(prob, queryVars);
       end
     end
  
     function [logZ] = lognormconst(obj, model)
       mu = model.mu; Sigma = model.Sigma;
      d = length(mu);
      logZ = (d/2)*log(2*pi) + 0.5*logdet(Sigma);
     end
    
    function [L] = logprob(obj, model, X, normalize)
    % L(i) = log p(X(i,:) | params)
        if nargin < 3, normalize = true; end
        mu = model.mu; Sigma = model.Sigma;
        if numel(mu)==1
            X = X(:); % ensure column vector
        end
        [N d] = size(X);
        if length(mu) ~= d
            error('X should be N x d')
        end
        if det(Sigma)==0
          L = repmat(NaN,N,1);
          return;
        end
        X = bsxfun(@minus,X,mu');
        L =-0.5*sum((X*inv(Sigma)).*X,2);
        if normalize
            L = L - lognormconst(obj, model);
        end
        if(0 && statsToolboxInstalled) % sanity check
            Lstats = log(mvnpdf(bsxfun(@plus,X,mu'),mu',Sigma));
            assert(approxeq(L,Lstats))
        end    
    end
    
  end % methods

end