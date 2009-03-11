classdef LinGaussHybridCPD < CondProbDist 
  % p(y|xc, xd) = N(y | wc'*xc + wd'*[xd-1] + w0, v) where xc is all the cts parents
  % and xd is all the discrete parents. (We subtract 1 from xd so state 1
  % codes as 0.)
  % This can be converted to a tabular factor if y and xc are observed.
  
  properties 
    wc;
    wd;
    w0;
    v; % variance
    dparentNdx;
    dparentArity;
  end
  
  methods
    function obj = LinGaussHybridCPD(w0, wc, wd,  v, dparentNdx, dparentArity)
      % dparentNdx specifies which of the parents are discrete
      % dparentArity(i) is the number of discrete states for discrete parent i
      if(nargin == 0),wc=[];wd=[];w0=[];v=[]; dparentNdx = []; dparentArity=[]; end
      obj.wc = wc; obj.wd = wd; obj.w0 = w0; obj.v = v;
      obj.dparentNdx = dparentNdx; obj.dparentArity = dparentArity;
    end
    
    
    function Tfac = convertToTabularFactor(CPD, domain, visVars, visVals)
      % domain = global indices of each parent, followed by index of child
      % visVars is in global numbering (relative to network, not CPD)
     child = domain(end); parents = domain(1:end-1);
     discreteParents = parents(CPD.dparentNdx);
     nparents = length(parents);
     cparentNdx = setdiffPMTK(1:nparents, CPD.dparentNdx);
     ctsParents = parents(cparentNdx);
     if ~isequal([ctsParents child], visVars)
       error('must observed all cts nodes')
     end
     valsFam = zeros(1,nparents+1);
     valsFam(lookupIndices(visVars, domain)) = visVals;
     ctsParentVals = valsFam(cparentNdx); 
     %ctsParentVals = valsFam(lookupIndices(ctsParents, domain));
     ctsChildVal = valsFam(end);
     xc = ctsParentVals; y = ctsChildVal;
     sigma = sqrt(CPD.v);
     nstates = prod(CPD.dparentArity);
     T = zeros(1, nstates);
     for i=1:nstates
       discreteParentVals = ind2subv(CPD.dparentArity, i);
       xd = discreteParentVals-1;
       mu = CPD.w0 + CPD.wc(:)'*xc(:) + CPD.wd(:)'*xd(:);
       T(i) = normpdf(y, mu, sigma);
     end
     T = reshapePMTK(T, CPD.dparentArity);
     Tfac = TabularFactor(T, discreteParents);
    end
    
  end
  
end
