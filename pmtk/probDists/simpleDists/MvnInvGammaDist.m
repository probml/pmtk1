classdef MvnInvGammaDist < ProbDist
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
            b = rowvec(b);
            %if(numel(b) == 1 && numel(mu) > 1), b = b*ones(1,numel(mu)); end;
            %if(length(b) ~= length(a)), a = a*ones(1,length(b)); end;
            m.mu = mu; m.Sigma = Sigma; m.a = a; m.b = b;
        end

        %{
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
%}
        
    		function m = mode(obj, varargin)
          [covtype] = processArgs(varargin, '-covtype', 'spherical');
					d = length(obj.mu);
   		   	% Returns a structure
		      m.mu = obj.mu;
      		% m.Sigma = obj.Sigma; % this may be the wrong formula...
          switch lower(covtype)
            case 'spherical'
      		    m.Sigma = diag(obj.b ./ (obj.a + 1/2*d + 1 ))*eye(d); % this should be the correct formula...
            case 'diagonal'
						  m.Sigma = diag(obj.b ./ (obj.a + 1/2 + 1));
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
          [mu, Sigma, cov] = processArgs(varargin, '-mu', [], '-Sigma', [], '-covtype', 'diagonal');
          if(isempty(mu) || isempty(Sigma))
            l = 0; return;
          end;
          mu = mu';
          if(size(Sigma,1) ~= size(mu,2))
            error('MvnInvGammaDist:logprob', 'Incorrect dimensions for mu, Sigma')
          end;
          if(size(Sigma,3) > 1 && size(mu,1) ~= size(Sigma,3))
            error('MvnInvGammaDist:logprob', 'Number of rows of mu must match number of pages of Sigma when more than one Sigma is passed in');
          end;
          [n,d] = size(mu);
          m0 = obj.mu; k = obj.Sigma; a = obj.a; b = obj.b;
          l = zeros(n,1);
          for i=1:n
            gausslogprob = -1/2*logdet(2*pi*Sigma(:,:,i)) - k/2*(mu(i,:) - m0')*inv(Sigma(:,:,i))*(mu(i,:) - m0')';
            invsigmalogprob = sum(b.*log(a) - gammaln(a) - (a+1).*rowvec(diag(Sigma(:,:,i))) - b./rowvec(diag(Sigma(:,:,i))));
          end
          l = gausslogprob + invsigmalogprob;
          %error('not yet implemented');
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
