classdef MvnInvWishartDist < ParamDist
  % p(m,S|params) = N(m|mu, 1/k * S) IW(S| dof, Sigma)
  properties
    mu;
    Sigma;
    dof;
    k;
  end

  %% main methods
  methods
    function m = MvnInvWishartDist(varargin)
      if nargin == 0, varargin = {}; end
      [mu, Sigma, dof, k] = process_options(...
        varargin, 'mu', 0, 'Sigma', 1, 'dof', 1, 'k', []);
      d = length(mu);
      if isempty(k), k = d; end
      m.mu = mu; m.Sigma = Sigma; m.dof = dof; m.k = k;
      if dof < d, warning('MvnInvWishartDist:dofSize','dof should be bigger than dimensionality'); end
    end
    
    function m = setParam(m, param)
      m.mu = param.mu;
      m.Sigma = param.Sigma;
      m.dof = param.dof;
      m.k = param.k;
    end

    function d = ndimensions(obj)
      % for the purposes of plotting etc, the dimensionality is |m|+|S|
      d = length(obj.mu) + length(obj.Sigma(:));
    end
    
    function lnZ = lognormconst(obj)
      d = length(obj.mu); 
      v = obj.dof;
      S = obj.Sigma;
      k = obj.k;
      lnZ = (v*d/2)*log(2) + mvtGammaln(d,v/2) -(v/2)*logdet(S) + (d/2)*log(2*pi/k);
    end

    function [m,S] = sample(obj,n)
      if (nargin < 2), n = 1; end;
      mu = obj.mu; k = obj.k;
      d = length(mu);
      m = zeros(n,d); S = zeros(d,d,n);
      SigmaDist = InvWishartDist(obj.dof, obj.Sigma);
      S = sample(SigmaDist,n);
      for s=1:n
        m(s,:) = sample(MvnDist(mu, S(:,:,s) / k),1);
      end
    end
     
    function L = logprob(obj, mu, Sigma)
        if(nargin == 2) % as in logprob(obj,X) used by plot
            % vectorized w.r.t. both mu and Sigma
            X = mu;
            mu = X(:,1);
            if(size(X,2) ~= 2)
                error('MvnInvWishartDist.logprob(obj,X) syntax only supported when X(:,1) = mu and X(:,2) = Sigma. Use logprob(obj,mu,Sigma) instead.');
            end
            piw = InvWishartDist(obj.dof, obj.Sigma);
            N = size(X,1);
            L = zeros(N,1);
            pgauss = MvnDist(obj.mu,[]);  % faster not to recreate the object each iteration
            k = obj.k;                    
            for i=1:N
               Sigma = X(i,2);
               pgauss.Sigma = Sigma/k; % invalid - X may be negative
               L(i) = logprob(pgauss,mu(i)) + logprob(piw,Sigma);
            end
        else
            pgauss = MvnDist(obj.mu, Sigma/obj.k);
            piw = InvWishartDist(obj.dof, obj.Sigma);
            L = logprob(pgauss, mu(:)') + logprob(piw, Sigma);
        end
    end
     
%      function plot(obj,varargin)
%         subplot(1,2,1);
%         plot(marginal(obj,'mu'),varargin{:});
%         xlabel('Marginal on ''mu''');
%         subplot(1,2,2);
%         plot(marginal(obj,'Sigma'),varargin{:});
%         xlabel('Marginal on ''Sigma''');
%      end
       
    %{
    function L = logprob(obj, X)
      % For 1d, L(i) = log p(X(i,:) | theta), where X(i,:) = [mu sigma]
      d = length(obj.mu);
      if d > 1, error('not supported'); end
      pgauss = MvnDist(obj.mu, obj.Sigma/obj.k); % obj.Sigma is the argument!!
      %warning('wrong formula')
      piw = InvWishartDist(obj.dof, obj.Sigma);
      n = size(X,1);
      assert(size(X,2)==2);
      L = logprob(pgauss, X(:,1)) + logprob(piw, X(:,2));
    end
    %}
      
    function m = mode(obj, varargin)
			d = size(obj.Sigma,1);
      % Returns a structure
      m.mu = obj.mu;
      % m.Sigma = obj.Sigma; % this may be the wrong formula...
      m.Sigma = obj.Sigma / (obj.dof + d + 2); % this should be the correct formula...
    end
    
    function mm = marginal(obj, queryVar)
      % marginal(obj, 'Sigma') or marginal(obj, 'mu')
      switch lower(queryVar)
        case 'sigma'
          d = size(obj.Sigma,1);
          if d==1 
            mm = InvGammaDist(obj.dof/2, obj.Sigma/2);
          else
            mm = InvWishartDist(obj.dof, obj.Sigma);
          end
        case 'mu'
          d = length(obj.mu);
          v = obj.dof;
          if d==1
            mm = StudentDist(v, obj.mu, obj.Sigma/(obj.k*v));
          else
            mm = MvtDist(v-d+1, obj.mu, obj.Sigma/(obj.k*(v-d+1)));
          end
        otherwise
          error(['unrecognized variable ' queryVar])
      end
    end
    
   
    

  end
  
  methods
      function xrange = plotRange(obj)
          d = length(obj.mu);
          if d > 1, error('not supported'); end
          sf = 2;
          S = obj.Sigma/obj.k;
          xrange = [obj.mu-sf*S, obj.mu+sf*S, 0.01, sf*S];
      end
  end
    
    
end
