classdef TrellisDist < NonParamDist
% A distribution over discrete paths
% 
% NOTATION: y(t) is the the t-th observation, length(y) is the length of the
%                observation.
%           S(t) is the t-th hidden, (discrete) state


    properties
        
        pi;   % The distribution over starting states
        A;    % A(i,j) = p(S(t+1)=j | S(t)=i) - the transition matrix: 
              % size(A) = nstates-by-nstates
              
        B;    % B(i,t) = p(y(t) | S(t)=i)     - the matrix of local evidence: 
              % size(B) = nstates-by-length(y)
            
    end
    
    properties(GetAccess = 'public', SetAccess = 'protected')
    % Values here are calculated only as needed and stored for future queries.   
    
    
      alpha;    % alpha(i,t) = p(S(t)=i| y(1:t))       (filtering) forwards step    
      beta;     % beta(i,t) propto p(y(t+1:T) | Z(t=i))            backwards step
      gamma;    % gamma(i,t) = p(S(t)=i | y(1:T))      (smoothing) forwards/backwards
      gamma2;
      xi;       % two slice marginal distribution 
                % xi(i,j,t)  = p(Z(t)=i, Z(t+1)=j | y(1:T)) , t=1:T-1
    end
    
    properties(GetAccess = 'protected' , SetAccess = 'protected')
                
      viterbiPath; % the most likely sequence of hidden states given model and 
                   % local evidence.
                   
      logp;        % The log probability of the local evidence             
                   
    end
    
    
    methods
        
        function trellis = TrellisDist(pi,A,B)
        % Construct a new trellis dist with the a distribution over starting 
        % states pi, a transition matrix A and a matrix of local evidence, B. 
            if(nargin > 0), trellis.pi = pi; end
            if(nargin > 1), trellis.A =  A ; end
            if(nargin > 2), trellis.B =  B ; end    
            
        end
        
        %% Setters
        function trellis = set.pi(trellis,pi)
            reset(trellis);
            trellis.pi = pi;
        end
        
        function trellis = set.A(trellis,A)
           reset(trellis);
           trellis.A = A;
        end
        
        function trellis = set.B(trellis,B)
           reset(trellis)
           trellis.B = B;
        end
    end
    
    methods
        function [path,trellis] = mode(trellis)
        % runs viterbi
            checkParams(trellis);
            if(isempty(trellis.viterbiPath))
                path = hmmViterbi(trellis.pi,trellis.A,trellis.B);
                trellis.viterbiPath = path;
            else
                path = trellis.viterbiPath; 
            end
        end
        
        function [m,trellis] = marginal(trellis,ti,tj,method)
        % Compute the one or two slice marginals. 
        %
        % Examples:
        %
        % [m,trellis] = marginal(trellis,ti,[],'smoothed') - return the ti-th
        % one-slice marginal, where the marginals are calculated using
        % forwards-backwards, (default). 
        %
        % [m,trellis] = marginal(trellis,ti,[],'filtered') - same as above except
        % the marginals are calculated using forwards only. 
        % 
        % [m,trellis] = marginal(trellis,ti,tj) - compute the two slice marginal
        % xi(ti,tj). 
        % 
        % [m,trellis] = marginal(trellis) - return all of the two slice
        %                                   marginals, i.e. xi where
        % xi_full(i,j,t)  = p(S(t)=i, S(t+1)=j | y(1:T)) , t=1:T-1
        % xi(i,j)         = sum_{t=1}^{T-1} xi_full(i,j,t) 
        %
        % [m,trellis] = marginal(trellis,':') return all of the one slice
        % marginals, (smoothed)
        %
        % [m,trellis] = marginal(trellis,':',[],'filtered') return all of the
        % one slice marginals (filtered)
       
            checkParams(trellis);
            if(nargin < 4)
                method = 'smoothed';
            end
            
            if(nargin < 2 || (nargin > 2 && ~isempty(tj)))
               if(isempty(trellis.xi))
                    if(isempty(trellis.alpha)||isempty(trellis.beta))
                        [trellis.gamma, trellis.alpha, trellis.beta, trellis.logp] = hmmFwdBack(trellis.pi, trellis.A, trellis.B);
                    end
                    trellis.xi = hmmComputeTwoSlice(trellis.alpha, trellis.beta, trellis.A, trellis.B);
               end
            end
            
            if(nargin < 2) % return the whole two slice dist
                m = trellis.xi;
                return;
            end
            
            if(isempty(tj)) % compute one slice
               switch lower(method)
                   case 'filtered'
                       if(isempty(trellis.alpha))
                           [trellis.gamma, trellis.alpha, trellis.beta, trellis.logp] = hmmFwdBack(trellis.pi, trellis.A, trellis.B);
                       end
                       m = trellis.alpha(:,ti);
                   case 'smoothed'
                       if(isempty(trellis.gamma))
                           [trellis.gamma, trellis.alpha, trellis.beta, trellis.logp] = hmmFwdBack(trellis.pi, trellis.A, trellis.B);
                       end
                       m = trellis.gamma(:,ti);
                   otherwise
                        error('%s is an invalid method',method);
               end
                
            else            % compute two slice
                m = trellis.xi(ti,tj);
            end
         
        end
        
        function [expWeights,trellis] = expectedWeights(trellis)
        % Compute the expected weigths that will be used along with the
        % observation in calculating the expected sufficient statistics.
            checkParams(trellis);
            if(isempty(trellis.gamma))
                [trellis.gamma, trellis.alpha, trellis.beta, trellis.logp] = hmmFwdBack(trellis.pi, trellis.A, trellis.B);
            end
            expWeights = sum(trellis.gamma,2);
        end
        
        function [logp,trellis] = logprob(trellis)
            
            if(isempty(trellis.logp))
                [trellis.alpha, trellis.logp] = hmmFilter(trellis.pi, trellis.A, trellis.B);
            end
            logp = trellis.logp;
            
        end
        
        function [s,trellis] = sample(trellis,s)
            checkParams(trellis);
        end
        
        
        function [pred,trellis] = predictFutureObservations(trellis,horizon)
            
        end
        
        function [h,trellis] = plot(trellis,varargin)
            checkParams(trellis);
        end
        
        function trellis =  reset(trellis)
        % Clears stored values used in lazy evaluation. This is called
        % automatically when the parameters are changed. 
          trellis.alpha = [];  
          trellis.beta = [];    
          trellis.gamma = [];   
          trellis.gamm2 = [];
          trellis.xi = [];      
          trellis.viterbiPath = []; 
          trellis.logp = [];  
        end
        
        function trellis = precompute(trellis)
        % Precompute values rather than rely on lazy evaluation.     
            [trellis.gamma, trellis.alpha, trellis.beta, trellis.logp] = hmmFwdBack(trellis.pi, trellis.A, trellis.B);
             trellis.xi = hmmComputeTwoSlice(trellis.alpha, trellis.beta, trellis.A, trellis.B);
           
        end
        
    end
    
    methods(Access = 'protected')
        
        function checkParams(trellis)
           if(isempty(trellis.pi) || isempty(trellis.A) || isempty(trellis.B))
              error('All of {pi,A,B} must be specified'); 
           end
            
        end
        
        
    end
    
end

