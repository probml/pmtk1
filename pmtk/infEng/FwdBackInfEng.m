classdef FwdBackInfEng < InfEng

% NOTATION: y(t) is the the t-th observation, length(y) is the length of the
%                observation.
%           S(t) is the t-th hidden, (discrete) state


    properties
        
        pi;   % The distribution over starting states
        A;    % A(i,j) = p(S(t+1)=j | S(t)=i) - the transition matrix: 
              % size(A) = nstates-by-nstates (Each *row* sums to one unlike the
              % DiscreteDist class)
              
        B;    % B(i,t) = p(y(t) | S(t)=i)     - the matrix of local evidence: 
              % size(B) = nstates-by-length(y)
            
    end
    
    properties(GetAccess = 'public', SetAccess = 'protected')
    % Values here are calculated only as needed and stored for future queries.   
    
    
      alpha;    % alpha(i,t) = p(S(t)=i| y(1:t))       (filtering) forwards step    
      beta;     % beta(i,t) propto p(y(t+1:T) | Z(t=i))            backwards step
      gamma;    % gamma(i,t) = p(S(t)=i | y(1:T))      (smoothing) forwards/backwards
      xi;       % two slice marginal distribution 
      
      viterbiPath; % the most likely sequence of hidden states given model and 
                   % local evidence.
                   
      logp;        % The log probability of the local evidence            
    
    end
    
   
    
    
    methods
        
        function eng = FwdBackInfEng(pi,A,B)
        % Construct a new eng with the a distribution over starting 
        % states pi, a transition matrix A and a matrix of local evidence, B. 
            if(nargin > 0)
                
                if(~isnumeric(pi))
                   pi = mean(pi); 
                end
                
                eng.pi = pi; 
                
            end
            if(nargin > 1)
               
                if(~isnumeric(A))
                    A = mean(A)';
                end
                eng.A =  A ;
            end
            if(nargin > 2), eng.B =  B ; end 
            
        end
        
        %% Setters
        function eng = set.pi(eng,pi)
            eng = reset(eng);
            eng.pi = rowvec(pi);
        end
        
        function eng = set.A(eng,A)
           eng = reset(eng);
           assert(approxeq(A,normalize(A,2)));
           eng.A = A;
        end
        
        function eng = set.B(eng,B)
           eng = reset(eng);
           eng.B = B;
        end
    end
    
    methods
        
        
        function eng = condition(eng,model,visVars,visValues)   
            eng = reset(eng);
            eng.pi = mean(model.startDist);
            eng.A = mean(model.transitionDist)';
            if(iscell(visVars))
                error('conditioning on multiple variables not yet supported');
            else
               if(isequal(visVars,'Y'))
                   eng.B = makeLocalEvidence(model,visValues);
               elseif(isequal(visVars,'Z'))
                   error('conditioning on observed values for the latent variables not yet supported');
               else
                   error('Valid variables are ''Y'' for the emission observations and ''Z'' for latent');
               end
            end
        end
        
        
        function [path,eng] = mode(eng)
        % runs viterbi
            checkParams(eng);
            if(isempty(eng.viterbiPath))
                path = hmmViterbi(eng.pi,eng.A,eng.B);
                eng.viterbiPath = path;
            else
                path = eng.viterbiPath; 
            end
        end
        
        function [m,eng] = marginal(eng,ti,tj,method)
        % Compute the one or two slice marginals. 
        %
        % Examples:
        %
        % [m,eng] = marginal(eng,ti,[],'smoothed') - return the ti-th
        % one-slice marginal, where the marginals are calculated using
        % forwards-backwards, (default). 
        %
        % [m,eng] = marginal(eng,ti,[],'filtered') - same as above except
        % the marginals are calculated using forwards only. 
        % 
        % [m,eng] = marginal(eng,ti,tj) - compute the two slice marginal
        % xi(ti,tj). 
        % 
        % [m,eng] = marginal(eng) - return all of the two slice
        %                                   marginals, i.e. xi where
        % xi_full(i,j,t)  = p(S(t)=i, S(t+1)=j | y(1:T)) , t=2:T
        % xi(i,j)         = sum_{t=2}^{T} xi_full(i,j,t) 
        %
        % [m,eng] = marginal(eng,':') return all of the one slice
        % marginals, (smoothed)
        %
        % [m,eng] = marginal(eng,':',[],'filtered') return all of the
        % one slice marginals (filtered)
       
            checkParams(eng);
            if(nargin < 4)
                method = 'smoothed';
            end
            
            if(nargin < 2 || (nargin > 2 && ~isempty(tj)))
               if(isempty(eng.xi))
                    if(isempty(eng.alpha)||isempty(eng.beta))
                        [eng.gamma, eng.alpha, eng.beta, eng.logp] = hmmFwdBack(eng.pi, eng.A, eng.B);
                    end
                    eng.xi = hmmComputeTwoSlice(eng.alpha, eng.beta, eng.A, eng.B);
               end
            end
            
            if(nargin < 2) % return the whole two slice dist
                m = eng.xi;
                return;
            end
            
            if(nargin < 3 || isempty(tj)) % compute one slice
               switch lower(method)
                   case 'filtered'
                       if(isempty(eng.alpha))
                           [eng.gamma, eng.alpha, eng.beta, eng.logp] = hmmFwdBack(eng.pi, eng.A, eng.B);
                       end
                       m = eng.alpha(:,ti);
                   case 'smoothed'
                       if(isempty(eng.gamma))
                           [eng.gamma, eng.alpha, eng.beta, eng.logp] = hmmFwdBack(eng.pi, eng.A, eng.B);
                       end
                       m = eng.gamma(:,ti);
                   otherwise
                        error('%s is an invalid method',method);
               end
                
            else            % compute two slice
                m = eng.xi(ti,tj);
            end
         
        end
        
      
        function [logp,eng] = logprob(eng)
            
            if(isempty(eng.logp))
                [eng.alpha, eng.logp] = hmmFilter(eng.pi, eng.A, eng.B);
            end
            logp = eng.logp;
            
        end
        
        function [s,eng] = sample(eng,nsamples)
            checkParams(eng);
            s = hmmSamplePost(eng.pi, eng.A, eng.B, nsamples);           
        end
                 
        function eng =  reset(eng)
        % Clears stored values used in lazy evaluation. This is called
        % automatically when the parameters are changed. 
          eng.alpha = [];  
          eng.beta = [];    
          eng.gamma = [];   
          eng.xi = [];      
          eng.viterbiPath = []; 
          eng.logp = [];  
        end
        
        function eng = precompute(eng)
        % Precompute values rather than rely on lazy evaluation.     
            [eng.gamma, eng.alpha, eng.beta, eng.logp] = hmmFwdBack(eng.pi, eng.A, eng.B);
             eng.xi = hmmComputeTwoSlice(eng.alpha, eng.beta, eng.A, eng.B);
           
        end
        
        function d = ndimensions(eng)
            d = size(eng.B,2);
        end
            
            
        
    end
    
    methods(Access = 'protected')
        
        function checkParams(eng)
           if(isempty(eng.pi) || isempty(eng.A) || isempty(eng.B))
              error('All of {pi,A,B} must be specified'); 
           end
            
        end
        
        
    end
    
end

