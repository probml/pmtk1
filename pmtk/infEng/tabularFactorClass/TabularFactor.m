classdef TabularFactor 
  % tabular (multi-dimensional array) factor/potential
  
  properties
    T;
    domain;
    sizes;
  end
  
  %% main methods
  methods
    function m = TabularFactor(T, domain)
      if nargin < 1, T = []; end
      if nargin < 2, domain = 1:myndims(T); end
      m.T = T;
      m.domain = domain;
      m.sizes = mysize(T);
    end

    function smallpot = marginalize(bigpot, onto, maximize)
      % smallpot = marginalizeFactor(bigpot, onto, maximize)
      if nargin < 3, maximize = 0; end
      ns = zeros(1, max(bigpot.domain));
      ns(bigpot.domain) = bigpot.sizes;
      smallT = marg_table(bigpot.T, bigpot.domain, bigpot.sizes, onto, maximize);
      smallpot = TabularFactor(smallT, onto);
    end
    
    function Tbig = multiplyBy(Tbig, Tsmall)
      % Tsmall's domain must be a subset of Tbig's domain.
      Ts = extend_domain_table(Tsmall.T, Tsmall.domain, Tsmall.sizes, Tbig.domain, Tbig.sizes);
      Tbig.T(:) = Tbig.T(:) .* Ts(:);  % must have bigT(:) on LHS to preserve shape
    end
    
    function Tbig = divideBy(Tbig, Tsmall)
      % Tsmall's domain must be a subset of Tbig's domain.
      Ts = extend_domain_table(Tsmall.T, Tsmall.domain, Tsmall.sizes, Tbig.domain, Tbig.sizes);
      % Replace 0s by 1s before dividing. This is valid, Ts(i)=0 iff Tbig(i)=0.
      Ts = Ts + (Ts==0);
      Tbig.T(:) = Tbig.T(:) ./ Ts(:);  % must have bigT(:) on LHS to preserve shape
    end
    
    function T = multiplyFactors(varargin)
      % T = multiplyFactors(fac1, fac2, ...)
      N = length(varargin);
      dom = [];
      for i=1:N
        Ti = varargin{i};
        dom = [dom Ti.domain];
      end
      dom = unique(dom);
      ns = zeros(1, max(dom));
      for i=1:N
        Ti = varargin{i};
        ns(Ti.domain) = Ti.sizes;
      end
      sz = prod(ns(dom));
      if sz>10000
       sprintf('creating tabular factor with %d entries', sz)
      end
      T = tabularFactor(myones(ns(dom)), dom);
      for i=1:N
        Ti = varargin{i};
        T = multiplyBy(T, Ti);
      end
    end

    function [Tfac, Z] = normalizeFactor(Tfac)
      [Tfac.T, Z] = normalize(Tfac.T);
    end
    
    function Fsmall = slice(Fbig, visNodes, visValues)
      % Return Tsmall(hnodes) = Tbig(visNodes=visValues, hnodes=:)
      if isempty(visNodes), Fsmall = Fbig; return; end
      vnodes = lookupIndices(visNodes, Fbig.domain);
      d = length(Fbig.sizes);
      %hnodes = setdiff(1:d, visNodes);
      hidNodes = mysetdiff(Fbig.domain, visNodes);
      hnodes = mysetdiff(1:d, vnodes);
      ndx = mk_multi_index(d, vnodes, visValues);
      TT = myreshape(Fbig.T, Fbig.sizes);
      Tsmall = squeeze(TT(ndx{:}));
      Fsmall = TabularFactor(Tsmall, hidNodes);
    end
    
    function x = mode(Tfac)
      x = argmax(Tfac.T);
    end
    
  end % methods

 

end