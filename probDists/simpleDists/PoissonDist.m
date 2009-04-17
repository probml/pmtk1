classdef PoissonDist < ParamDist
  
  properties
    lambda;
    support;
  end
 
  
  methods 
    function obj =  PoissonDist(lambda)
      if nargin == 0;
        lambda = [];
      end
      obj.lambda = lambda;
      obj.support = 0:25; % truncate support for plotting purposes
    end

    function d = ndimensions(obj)
      d = length(obj.lambda);
    end
    
    function m = mean(obj)
     checkParamsAreConst(obj)
      m = obj.lambda;
    end
    
    function m = mode(obj)
      checkParamsAreConst(obj)
      m = floor( obj.lambda );
    end  
    
    function m = var(obj)
      checkParamsAreConst(obj)
       m = obj.lambda;
    end
     
   
    function X = sample(obj, n)
       % X(i,j) = sample from params(j) for i=1:n
       checkParamsAreConst(obj)
       d = ndimensions(obj);
       X = zeros(n, d);
       if ~statsToolboxInstalled, error('need stats toolbox'); end
       for j=1:d
         X(:,j) = poissrnd(obj.lambda(j), n, 1);
       end
     end
    
     function [L,Lij] = logprob(obj, X)
        % L(i) = sum_j logprob(X(i,j) | params(j))
        % Lij(i,j) = logprob(X(i,j) | params(j))
       checkParamsAreConst(obj)
       d = ndimensions(obj);
       n = size(X,2);
       Lij = zeros(n,d);
       for j=1:d
         x = X(:,j);
         Lij(:,j) = x .* log(obj.lambda(j)) - factorialln(x) - obj.lambda(j);
       end
       L = sum(Lij,2);
     end


     function logZ = lognormconst(obj)
       logZ = -obj.lambda;
     end
        
     function h=plot(obj, varargin)
         % plot a probability mass function as a histogram
         % handle = plot(pmf, 'name1', val1, 'name2', val2, ...)
         % Arguments are
         % plotArgs - args to pass to the plotting routine, default {}
         %
         % eg. plot(p,  'plotArgs', 'r')
         [plotArgs] = process_options(...
             varargin, 'plotArgs' ,{});
         if ~iscell(plotArgs), plotArgs = {plotArgs}; end
         mu = rowvec(exp(logprob(obj,obj.support)));
         h=bar(mu, plotArgs{:});
         set(gca,'xticklabel',obj.support);
    end
     
     

  end
  %% Private methods
  methods(Access = 'protected')

    function checkParamsAreConst(obj)
      p = isa(obj.lambda, 'double');
      if ~p
        error('parameters must be constants')
      end
    end

  end
  
end