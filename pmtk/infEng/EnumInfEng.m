classdef EnumInfEng < InfEng & TableJointDist
  % inference by brute-force enumeration
  % in a tabular (multi-dimensional array) distribution
  
  properties
  end

  
  %% main methods
  methods
  
    function m = EnumInfEng(T, domain)
      m.T  = T;
      if nargin < 2, domain = 1:ndims(T); end
      m.domain = domain;
    end

  end
end

 