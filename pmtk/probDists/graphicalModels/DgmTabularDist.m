classdef DgmTabularDist < DgmDist 
  % directed  graphical model with discrete CPDs
  
  properties
    %CPDs;
    % infMethod in GM class
    % G field is in parent class
  end

  
  %%  Main methods
  methods
   
     function obj = DgmTabularDist(G, varargin)
       error('deprecated');
      if(nargin == 0);G = [];end
      if isa(G,'double'), G=Dag(G); end
      obj.G = G;
      [obj.CPDs, obj.infMethod, obj.domain]= process_options(...
        varargin, 'CPDs', [], 'infMethod', 'varElim', 'domain',[]);
      if isempty(obj.domain)
        obj.domain = 1:numel(obj.CPDs);
      end
     end
    
    function dgm = mkRndParams(dgm, varargin)
      d = ndimensions(dgm);
      for j=1:d
        pa = parents(dgm.G, j);
        q  = length(pa);
        dgm.CPDs{j} = TabularCPD(mkStochastic(rand(arity([pa j]))));
      end
    end
   
    
  end



end


