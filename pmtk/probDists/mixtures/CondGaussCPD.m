classdef CondGaussCPD < MixMvn
% Cts node Y with discrete parent Z, where p(Y|Z=k) = MVN(mu(:,k), Sigma(:,:,k))   
    
    properties
    end
    
    methods
      %{
      function CPD = CondGaussCPD(mu, Sigma)
        CPD.mu = mu;
        CPD.Sigma = Sigma;
      end
      %}
      function CPD = CondGaussCPD(varargin)
        if nargin == 0; return; end
        [CPD.distributions] = processArgs(varargin, '-distributions', []);
        nmix = length(CPD.distributions);
        CPD.mixingDistrib = DiscreteDist('T',normalize(ones(nmix,1)));
      end
      
      function Tfac = convertToTabularFactor(model, child, ctsParents, dParents, visible, data, nstates,fullDomain)
        %function Tfac = convertToTabularFactor(model, domain,visVars,visVals)
        % domain = indices of each parent, followed by index of child
        % all of the children must be observed
        assert(isempty(ctsParents))
        assert(length(dParents)==1)
        map = @(x)canonizeLabels(x,fullDomain);
        if visible(map(child))
          T = exp(calcResponsibilities(model,data(map(child))));
          Tfac = TabularFactor(T,dParents);
        else
          % barren leaf removal
          Tfac = TabularFactor(ones(1,nstates(map(dParents))), dParents);
        end
      end
      
        %{
function Tfac = convertToTabularFactor(model, child, ctsParents, dParents, visible, data, nstates);
%function Tfac = convertToTabularFactor(model, domain,visVars,visVals)
% domain = indices of each parent, followed by index of child
% all of the children must be observed
assert(isempty(ctsParents))
assert(length(dParents)==1)
assert(visible(child))
visVals = data(child);
if(isempty(visVars))
Tfac = TabularFactor(1,domain); return; % return an empty TabularFactor
end
pdom = domain(1); cdom = domain(2:end);
if ~isequal(cdom,visVars)
% If we have a mixture of factored bernoullis
% the factor would be all discrete, but we don't handle this
% case.
error('Not all of the children of this CPD were observed.');
end
T = exp(calcResponsibilities(model,visVals));
Tfac = TabularFactor(T,pdom); % only a factor of the parent now
end
    %}
      
      function p = isDiscrete(CPD) %#ok
        p = false;
      end

      function q = nstates(CPD)  %#ok
        %q = length(CPD.mixingWeights);
        q = length(CPD.distributions);
      end

    end
    
end