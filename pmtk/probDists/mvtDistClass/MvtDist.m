classdef MvtDist < ParamDist 
  % multivariate student T p(X|dof, mu,Sigma) 
  
  properties
    mu;
    Sigma;
    dof;
  end
  
  %% Main methods
  methods
    function m = MvtDist(dof, mu, Sigma)
      if nargin == 0
        mu = []; Sigma = []; dof = [];
      end
      m.mu  = mu;
      m.dof = dof;
      m.Sigma = Sigma;
    end
    
    function d = ndims(obj)
       d = numel(obj.mu); 
    end
    
    function objS = convertToScalarDist(obj)
      if ndims(obj) ~= 1, error('cannot convert to scalarDst'); end
      objS = StudentDist(obj.dof, obj.mu, obj.Sigma);
    end
    
    function logZ = lognormconst(obj)
      d = ndims(obj);
      v = obj.dof;
      logZ = -gammaln(v/2 + d/2) + gammaln(v/2) + 0.5*logdet(obj.Sigma) ...
        + (d/2)*log(v) + (d/2)*log(pi);
    end
    
    
    function L = logprob(obj, X)
      % L(i) = log p(X(i,:) | params)
      mu = obj.mu(:)'; % ensure row vector
      if length(mu)==1
        X = X(:); % ensure column vector
      end
      [N d] = size(X);
      if length(mu) ~= d
        error('X should be N x d')
      end
      %if statsToolboxInstalled
      %  L1 = log(mvtpdf(X-repmat(obj.mu,N,1), obj.Sigma, obj.dof));
      M = repmat(mu, N, 1); % replicate the mean across rows
      if isnan(obj.Sigma) | obj.Sigma==0
        L = repmat(NaN, N, 1);
      else
        mahal = sum(((X-M)*inv(obj.Sigma)).*(X-M),2);
        v = obj.dof;
        L = -0.5*(v+d)*log(1+(1/v)*mahal) - lognormconst(obj);
        %assert(approxeq(L,L1)) % fails unless Sigma is I
      end
    end
    

    
   
     function X = sample(m, n)
      % X(i,:) = sample for i=1:n
      checkParamsAreConst(obj)
      if statstoolboxInstalled
        X = mvtrnd(m.Sigma, m.dof, n) + repmat(m.mu,n,1);
      else
        R = chol(m.Sigma);
        X = repmat(mean(m)', n, 1) + (R'*trnd(m.dof, d, n))';
      end
    end

    function mu = mean(obj)
      checkParamsAreConst(obj)
      mu = obj.mu;
    end

    function mu = mode(obj)
      checkParamsAreConst(obj)
      mu = mean(obj);
    end

    function C = cov(obj)
      checkParamsAreConst(obj)
      C = (obj.dof/(obj.dof-2))*obj.Sigma;
    end
   
  
     
    function mm = marginal(obj, queryVars)
      % p(Q)
      checkParamsAreConst(obj)
      d = ndims(obj);
      if d == 1, error('cannot compute marginal of a 1d rv'); end
      mu = mean(obj); C = cov(obj);
      dims = queryVars;
      mm = MvtDist(obj.dof, mu(dims), C(dims,dims));
      if length(dims)==1
        mm = convertToScalarDist(mm);
      end
    end

    function mm = conditional(m, visVars, visValues)
      % p(Xh|Xvis=vis) 
      checkParamsAreConst(m)
      d = ndims(obj);
      if d == 1, error('cannot compute conditional of a 1d rv'); end
      % p(Xa|Xb=b)
      b = visVars; a = setdiff(1:d, b);
      dA = length(a); dB = length(b);
      if isempty(a)
        muAgivenB = []; SigmaAgivenB  = [];
      else
        mu = m.mu(:);
        xb = visValues;
        SAA = Sigma(a,a); SAB = Sigma(a,b); SBB = Sigma(b,b);
        SBBinv = inv(SBB);
        muAgivenB = mu(a) + SAB*SBBinv*(xb-mu(b));
        h = 1/(m.dof+dB) * (m.dof + (xb-muB)'*SBBinv*(xb-mu(b)));
        SigmaAgivenB = h*(SAA - SAB*SBBinv*SAB');
      end
      mm = MvtDist(m.dof + dA, muAgivenB, SigmaAgivenB);
    end
    
    function xrange = plotRange(obj, sf)
        if nargin < 2, sf = 3; end
        %if ndims(obj) ~= 2, error('can only plot in 2d'); end
        mu = mean(obj); C = cov(obj);
        s1 = sqrt(C(1,1));
        x1min = mu(1)-sf*s1;   x1max = mu(1)+sf*s1;
        if ndims(obj)==2
            s2 = sqrt(C(2,2));
            x2min = mu(2)-sf*s2; x2max = mu(2)+sf*s2;
            xrange = [x1min x1max x2min x2max];
        else
            xrange = [x1min x1max];
        end
    end

   
  end % methods


  %% Private methods
  methods(Access = 'protected')
    function checkParamsAreConst(obj)
      p = isa(obj.mu, 'double') && isa(obj.Sigma, 'double') && isa(obj.dof, 'double');
      if ~p
        error('params must be constant')
      end
    end
  end
  
end