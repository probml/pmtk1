classdef TrellisDist < NonParamDist
% A distribution over discrete paths

    properties
        pi;   % The distribution over starting states
        A;    % The transition matrix
        B;    % The matrix of local evidence
       
    end
    
    properties(GetAccess = 'protected', SetAccess = 'protected')
    % Values here are calculated only as needed and stored for future queries.   
      alpha;    % alpha(i,t) = p(Z(t)=i| y(1:t))       (filtering) forwards step    
      beta;     % beta(i,t) propto p(y(t+1:T) | Z(t=i))            backwards step
      gamma;    % gamma(i,t) = p(Z(t)=i | y(1:T))      (smoothing) forwards/backwards
      xi;       % two slice marginal distribution 
                % xi(i,j,t)  = p(Z(t)=i, Z(t+1)=j | y(1:T)) , t=1:T-1
    end
    
    
    methods
        
        function trellis = TrellisDist(pi,A,B)
            if(nargin > 0), trellis.pi = pi; end
            if(nargin > 1), trellis.A =  A ; end
            if(nargin > 2), trellis.B =  B ; end
            
            
        end
        
        function path = mode(trellis)
        % runs viterbi
            
        end
        
        function m = marginal(trellis,t)
        % computes one or two slice marginals - lazy evaluation, stores results
        % first time this is called. 
            
            
        end
        
        function e = expectation(trellis,f)
            
        end
        
        function s = sample(trellis,s)
            
        end
        
        function h = plot(trellis,varargin)
            
        end
        
        
        
        
    end
    
end

