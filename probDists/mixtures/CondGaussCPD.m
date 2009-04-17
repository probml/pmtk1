classdef CondGaussCPD
% Cts node Y with discrete parent Z, where p(Y|Z=k) = MVN(mu(:,k), Sigma(:,:,k))   
    
    properties
      mu;
      Sigma;
    end
    
    methods
      function CPD = CondGaussCPD(mu, Sigma)
        CPD.mu = mu;
        CPD.Sigma = Sigma;
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
      
      function p = isDiscrete(CPD) %#ok
        p = false;
      end

      function q = nstates(CPD)  %#ok
        q = length(CPD.mixingWeights);
      end

    end
    
end