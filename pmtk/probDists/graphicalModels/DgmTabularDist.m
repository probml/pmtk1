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
   
     function M = convertToUgm(obj, visVars, visVals)
       if(nargin < 2), visVars = [];  visVals = {}; end
      [Tfacs, nstates] = convertToTabularFactors(obj, visVars, visVals);
      M = UgmTabularDist('factors', Tfacs, 'nstates', nstates);
     end
    
    function [Tfacs, nstates] = convertToTabularFactors(obj,visVars,visVals)
    if(nargin < 2), visVars = [];  visVals = {}; end
    d = length(obj.CPDs);
    Tfacs = cell(1,d);
    nstates = zeros(1,d);
    for j=1:d
      dom = [parents(obj.G, j), j];
      include = ismember(visVars,dom);
      if(isempty(include) || ~any(include))
        Tfacs{j} = convertToTabularFactor(obj.CPDs{j}, dom, [],{});
      else
        Tfacs{j} = convertToTabularFactor(obj.CPDs{j}, dom, visVars(include) , visVals(include));
      end
      nstates(j) = Tfacs{j}.sizes(end);
    end
    end
    
    function Tfac = convertToJointTabularFactor(obj,visVars,visVals)
      % Represent the joint distribution as a single large factor
      if nargin < 3
        visVars = []; visVals = [];
      end
      Tfac = TabularFactor.multiplyFactors(convertToTabularFactors(obj,visVars,visVals));
    end
  end



end


