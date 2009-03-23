classdef LinGaussHybridCPD < CondProbDist 
  % p(y|xc, xd) = N(y | w0 + wc'*xc + wd'*[xd-1], v) where xc is all the cts parents
  % and xd is all the discrete parents. (We subtract 1 from xd so state 1
  % codes as 0.) This is linear regression where we treat categorical
  % variables as cts inputs.
  % This can be converted to a tabular factor if y and xc are observed.
  
  properties 
    wc;
    wd;
    w0;
    v; % variance
  end
  
  methods
    function obj = LinGaussHybridCPD(w0, wc, wd,  v)
      if(nargin == 0),wc=[];wd=[];w0=[];v=[]; end
      obj.wc = wc; obj.wd = wd; obj.w0 = w0; obj.v = v;
    end
    
    
      function p = isDiscrete(CPD) %#ok
        p = false;
      end
      
      function q = nstates(CPD)  %#ok
        q = 1;
      end
    
        
      function Tfac = convertToTabularFactor(CPD, child, ctsParents, dParents, visible, data, nstates,fullDomain)
          if ~visible(child) || ~all(visible(ctsParents))
              error('must observed all cts nodes')
          end
          if any(visible(dParents))
              error('currently we assume all discrete parents are hidden')
          end
          map = @(x)canonizeLabels(x,fullDomain);
          y = data(map(child));
          xc = data(map(ctsParents));
          sigma = sqrt(CPD.v);
          K = prod(nstates(map(dParents)));
          T = zeros(1, K);
          for i=1:K
              discreteParentVals = ind2subv(nstates(map(dParents)), i);
              xd = discreteParentVals-1;
              mu = CPD.w0 + CPD.wc(:)'*xc(:) + CPD.wd(:)'*xd(:);
              T(i) = normpdf(y, mu, sigma);
          end
          T = reshapePMTK(T, nstates(map(dParents)));
          Tfac = TabularFactor(T, dParents);
      end

    %{
    function Tfac = convertToTabularFactor(CPD, child, childVal, dparents, dparentVals, cparents, cparentVals, ...
        nstatesChild, nstatesCparents, nstatesDparents) %#ok
     if isempty(childVals) || isempty(cparentVals)
       error('must observed all cts nodes')
     end
     y = childVal;
     xc = cparentVals;
     sigma = sqrt(CPD.v);
     nstates = prod(nstatesDparents);
     T = zeros(1, nstates);
     for i=1:nstates
       discreteParentVals = ind2subv(nstatesDparents, i);
       xd = discreteParentVals-1;
       mu = CPD.w0 + CPD.wc(:)'*xc(:) + CPD.wd(:)'*xd(:);
       T(i) = normpdf(y, mu, sigma);
     end
     T = reshapePMTK(T, nstatesDparents);
     Tfac = TabularFactor(T, dparents);
    end
    %}
    
    %{
    function Tfac = convertToTabularFactor(CPD, domain, visVars, visVals)
      % domain = global indices of each parent, followed by index of child
      % visVars is in global numbering (relative to network, not CPD)
     child = domain(end); parents = domain(1:end-1);
     %discreteParents = parents(CPD.dparentNdx);
     discreteParents = CPD.dparentNdx;
     nparents = length(parents);
     %cparentNdx = setdiffPMTK(1:nparents, CPD.dparentNdx);
     ctsParents = setdiffPMTK(parents, discreteParents);
     %ctsParents = parents(cparentNdx);
     cparentNdx = lookupIndices(ctsParents, parents);
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
    %}
    
  end % methods
  
end
