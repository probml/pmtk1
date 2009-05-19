classdef SampleDist < ProbDist
  % Sample based representation of a pdf 
  % samples(1,s) is sample s (scalar)
  % samples(j,s) is sample s, j'th dimension  (vector)
  % samples(j,k,s)  (matrix)
  % weights(s) is a row vector
  
  properties
    samples; 
    weights;
  end

  %%  Main methods
  methods
    function m = SampleDist(X, weights)
      if nargin < 1, X = []; end
      if nargin < 2, weights = []; end
      nd = ndimsPMTK(X);
      switch nd
        case 1, m.samples = X(:)'; ns = length(X);
        otherwise, m.samples = X; ns = size(X,nd);
      end
      %m.samples = X;
      if isempty(weights)
        weights = (1/ns)*ones(1, ns);
      end
      m.weights = weights;
    end
    
    function mu = mean(obj)
      %nd = ndimsPMTK(obj.samples);
      %mu = mean(obj.samples, nd); % take mean across last dim
      w = obj.weights(:)';
      sz = size(obj.samples);
      ns = length(w);
      switch ndimsPMTK(obj.samples)
        case 1, mu = mean(w .* obj.samples);
        case 2, mu = mean(repmat(w, sz(1), 1) .* obj.samples,2);
        case 3, mu = mean(repmat(reshape(w,[1,1,ns]), [sz(1) sz(2) 1]) .* obj.samples,3);
        otherwise
          error('too many dims')
      end
    end
    
    
    function s = sample(obj,n)
      ndx = sampleDiscrete(obj.weights, 1, n);
      switch ndimsPMTK(obj.samples)
        case 1, s = obj.samples(ndx);
        case 2, s = obj.samples(:,ndx);
        case 3, s = obj.samples(:,:,ndx);
        otherwise
          error('too many dims')
      end
    end
   
    function L = logpdf(model, X)
      nd = ndimsPMTK(model.samples);
      if nd ~= 1
        error('can only compute logpdf of discrete scalar samples')
      end
      n = size(X,1);
      L = -inf(n,1);
      for i=1:n
        ppi = sum(model.weights(model.samples==X(i)));
        if ppi>0
        L(i) = log(ppi);
        end
      end
    end
  
  end % methods
  
  methods(Static = true)
    function test()
      setSeed(0);
      x1 = [1 2 3 4 5];
      w1 = normalize(1:5);
      s1 = SampleDist(x1,w1);
      s11 = sample(s1,1000);
      h=hist(s11, 1:5);
      assert(approxeq(w1, normalize(h), 1e-1))
      m1 = mean(s1);
      L = logpdf(s1, [1 2 6]');
      
      x2 = rand(5,10); 
      s2 = SampleDist(x2);
      s22 = sample(s2,10);
      m2 = mean(s2);
      
      x3 = rand(5,6,10);
      s3 = SampleDist(x3);
      s33 = sample(s3,20);
      m3 = mean(s3);
    end
  end
    
end

