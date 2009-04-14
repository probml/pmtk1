classdef MvnInvGammaDist < ParamDist
    % p(m,s2|params) = N(m|mu, s2 Sigma) IG(s2| a,b)
    properties
        mu;
        Sigma;
        a;
        b;
    end
    
    %% main methods
    methods
        function m = MvnInvGammaDist(varargin)
            if nargin == 0, varargin = {}; end
            [mu, Sigma, a, b] = process_options(...
                varargin, 'mu', [], 'Sigma', [], 'a', [], 'b', []);
            m.mu = mu; m.Sigma = Sigma; m.a = a; m.b = colvec(b);
        end

        function m = setParam(m, param)
          m.mu = param.mu;
          m.Sigma = param.Sigma;
          m.a = param.a;
          m.b = param.b;
        end

        function m = updateParam(m, param)
          m.mu = param.mu;
          m.Sigma = param.Sigma;
          m.a = param.a;
          m.b = param.b;
        end

    		function m = mode(obj)
					d = length(obj.mu);
   		   	% Returns a structure
		      m.mu = obj.mu;
      		% m.Sigma = obj.Sigma; % this may be the wrong formula...
					if(length(obj.b) == 1) % the case of the spherical covariance
	      		m.Sigma = (obj.b / (obj.a + 1/2*d + 1 )) * eye(d); % this should be the correct formula...
					else % the case of the diagonal covariance
						m.Sigma = diag(obj.b / (obj.a + 1/2 + 1));
					end
    		end
        
        
        function mm = marginal(obj, queryVar)
            % marginal(obj, 'sigma') or marginal(obj, 'mu')
            switch lower(queryVar)
                case 'sigma'
                    mm = InvGammaDist(obj.a, obj.b);
                case 'mu'
                    v = 2*obj.a;
                    s2 = 2*obj.b/v;
                    mm = MvtDist(v, obj.mu, s2*obj.Sigma);
                otherwise
                    error(['unrecognized variable ' queryVar])
            end
        end
        
        
        
        function d = ndimensions(obj)
            d = numel(obj.mu);
        end
        
        
        
        function l = logprob(obj,varargin)
            error('not yet implemented');
        end
        
        function xrange = plotRange(obj, sf)
            if nargin < 2, sf = 3; end
            %if ndimensions(obj) ~= 2, error('can only plot in 2d'); end
            mu = mean(obj); C = cov(obj);
            s1 = sqrt(C(1,1));
            x1min = mu(1)-sf*s1;   x1max = mu(1)+sf*s1;
            if ndimensions(obj)==2
                s2 = sqrt(C(2,2));
                x2min = mu(2)-sf*s2; x2max = mu(2)+sf*s2;
                xrange = [x1min x1max x2min x2max];
            else
                xrange = [x1min x1max];
            end
        end
        
        
    end
    
 
    
end
