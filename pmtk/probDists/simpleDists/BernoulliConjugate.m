classdef BernoulliConjugate < BinomConjugate
 % p(X,theta|a,b) = Ber(X|theta) Beta(theta|a,b) 
  
 
  methods 
    function obj =  BernoulliConjugate(varargin)
      [obj.muDist, obj.productDist] = processArgs(varargin, ...
        '-prior',NoPrior, '-productDist', false);
      obj.N = 1;
    end
           
  end 
 
end