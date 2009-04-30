classdef Bernoulli_BetaDist < Binom_BetaDist
 % p(X,theta|a,b) = Ber(X|theta) Beta(theta|a,b) 
  
 
  methods 
    function obj =  Bernoulli_BetaDist(varargin)
      [obj.muDist, obj.productDist] = processArgs(varargin, ...
        '-prior', [], '-productDist', false);
      obj.N = 1;
    end
           
  end 
 
end