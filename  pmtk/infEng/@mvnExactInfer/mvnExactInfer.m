classdef mvnExactInfer < infEngine
  % exact inference in the multivariate normal 
  
  properties
    ndims;
    model; % model params
    bel; % belief state - gets updated with evidence
    visVars = [];
    visValues = [];
    evidenceEntered = false;
  end
 
  methods
    function eng = mvnExactInfer(mu, Sigma)
      if nargin == 0, mu = []; Sigma = []; end
      eng = setParams(eng, {mu, Sigma});
    end
    
    function eng = setParams(eng, params)
      eng.model.mu = params{1};
      eng.model.Sigma = params{2};
      eng.ndims = length(eng.model.mu);
    end
    %{
    function eng = setParams(eng, mu, Sigma)
      eng.model.mu = mu; eng.model.Sigma = Sigma;
      eng.ndims = length(mu);
    end
    %}
     function [X, eng] = sample(eng, n)
       % X(i,:) = sample for i=1:n
       if ~eng.evidenceEntered
         eng = enterEvidence(eng, [], []); % compute joint
       end
       mu = eng.bel.mu; C = eng.bel.Sigma; d = length(mu);
       if statsToolboxInstalled
         X = mvnrnd(mu(:)', C, n);
       else
         R = chol(C);
         X = repmat(mu', n, 1) + (R'*randn(d,n))';
       end
     end
    
    function [postQuery, eng] = marginal(eng, queryVars)
      if ~eng.evidenceEntered
        eng = enterEvidence(eng, [], []);  % compute joint
      end
      d = eng.ndims;
      %[V,H] = findVisHid(eng.x);
      V = eng.visVars; 
      H = mysetdiff(1:d, V);
      mu = zeros(d,1);
      Sigma = zeros(d,d);
      xV = eng.visValues;
      mu(V) = xV; % re-insert visible data
      mu(H) = eng.bel.mu; % we only store distribution of hidden variables
      Sigma(H,H) = eng.bel.Sigma; 
      dims = queryVars;
      postQuery = mvnDist(mu(dims), Sigma(dims,dims));
    end


   function eng = enterEvidence(eng, visVars, visValues)
      % Set state to p(Xh|Xvis=vis)
      %d = eng.ndims;
      % enter data into current belief state to allow recursive updating
      % but indexing might be wrong
      %[muAgivenB, SigmaAgivenB] = gaussianConditioning(...
      %  eng.bel.mu, eng.bel.Sigma, visVars, visValues);
       % enter data into fresh belief state
      [muAgivenB, SigmaAgivenB] = gaussianConditioning(...
        eng.model.mu, eng.model.Sigma, visVars, visValues);
      eng.bel.mu = muAgivenB; eng.bel.Sigma = SigmaAgivenB;
      eng.visVars = visVars; eng.visValues = visValues;
      eng.evidenceEntered = true;
   end
   
  
  
  end % methods

end