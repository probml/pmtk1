classdef ParamDist < ProbDist
    % Parametric distribution
    
    methods
        
      function d = ndimensions(obj)
        % By default, we asssume the distribution is over a scalar rv
        % If the class defines a vector rv, it should over-ride this
        % method.
        d = 1;
      end
      
      function d = ndistrib(obj)
        % By default, we asssume the distribution is a single distribution
        % not a product/set.
        d = 1;
      end
      
     
    end
    
end

