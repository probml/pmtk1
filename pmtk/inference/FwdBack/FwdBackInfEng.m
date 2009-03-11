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
    
    
      alpha;     % alpha(i,t) = p(S(t)=i| y(1:t))       (filtering) forwards step    
      beta;      % beta(i,t) propto p(y(t+1:T) | Z(t=i))            backwards step
      gamma;     % gamma(i,t) = p(S(t)=i | y(1:T))      (smoothing) forwards/backwards
      xi_summed; % xi_summed(i,j) = sum_{t=2}^{T} xi(i,j,t) 
      xi;        % xi(i,j,t)  = p(S(t)=i, S(t+1)=j | y(1:T)) , t=2:T
      
      viterbiPath; % the most likely sequence of hidden states given model and 
                   % local evidence.
                   
      logp;        % The log probability of the local evidence            
      model;       % The model using this inference engine. 
    
    end
    
   
    
    
    methods
        
        function eng = FwdBackInfEng(pi,A,B)
        % Construct a new eng with the a distribution over starting 
        % states pi, a transition matrix A and a matrix of local evidence, B. 
            if(nargin > 0)
                if(~isnumeric(pi))
                   pi = mean(pi)'; 
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
            eng.pi = mean(model.startDist)';
            eng.A  = mean(model.transitionDist)';
            model.infEng = []; % avoid recursive copying
            eng.model = model;
            if(nargin < 4)
                visVars = []; visValues = [];
            end
            if(iscell(visVars))
                error('conditioning on multiple variables not yet supported');
            else
               if(isequal(visVars,'Y')||isempty(visVars))
                   if(nobservations(model,visValues) > 1)
                      error('You can only condition on one observation sequence at a time'); 
                   end
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
                [path,j,j] = hmmViterbi(log(eng.pi),log(eng.A),log(eng.B));
                eng.viterbiPath = path;
            else
                path = eng.viterbiPath; 
            end
        end


            function [m,eng] = marginal(eng,onto,method)
            % Compute marginals with respect to the latent discrete variables. 
            % 
            % marginal(eng) = xi_summed where xi_summed(i,j) = sum_{t=2}^{T}(xi(i,j,t)) - summed two slice marginals usefull for estimating the transition matrix
            % 
            % marginal(eng,':') = gamma where gamma(i,t) = p(S(t)=i | y(1:T)) 
            %
            % marginal(eng,':','filtered') = alpha, where alpha(i,t) = p(S(t)=i|y(1:t)) (forwards only)
            %
            % marginal(eng,t) = one slice marginal at time t
            %
            % marginal(eng,[t,t+1]) = two slice marginal onto [t,t+1]
            %
            % marginal(eng,[a,b,c,...]) - arbitrary marginal computed using
            % variable elimination.
                checkParams(eng);
                if nargin < 3
                    method = 'smoothed';
                end
                
                %% Return xi_summed if call was marginal(eng)
                if nargin == 1
                   if isempty(eng.xi_summed)
                       if isempty(eng.alpha) || isempty(eng.beta)
                           [eng.gamma, eng.alpha, eng.beta, eng.logp] = hmmFwdBack(eng.pi, eng.A, eng.B);
                       end
                      eng.xi_summed = hmmComputeTwoSlice(eng.alpha, eng.beta, eng.A, eng.B);
                   end
                    m = eng.xi_summed;
                   return; 
                end
                %% 
                % Pad local evidence with ones if horizon is beyond the time
                % frame of the observed data. This is valid since the
                % probability of an empty event is 1. 
                if ~ischar(onto)
                    horizon = max(onto);
                    if(size(eng.B,2) < horizon)
                        eng.B = [eng.B,ones(size(eng.A,1),horizon-size(eng.B,2))];
                    end
                end
                %% Return one slice marginals
                if numel(onto) == 1
                    switch lower(method)
                        case 'filtered'
                            if isempty(eng.alpha)
                                [eng.alpha, eng.logp] = hmmFilter(eng.pi, eng.A, eng.B);
                            end
                            m = eng.alpha(:,onto);
                        case 'smoothed'
                            if isempty(eng.gamma)
                                [eng.gamma, eng.alpha, eng.beta, eng.logp] = hmmFwdBack(eng.pi, eng.A, eng.B);
                            end
                            m = eng.gamma(:,onto);
                        otherwise
                            error('%s is an invalid method',method);
                    end
                    return;
                end
                %% Return two slice marginals from xi
                if (numel(onto) == 2) && ((onto(2)-onto(1)) == 1)
                    if isempty(eng.xi)
                        if isempty(eng.alpha) || isempty(eng.beta)
                           [eng.gamma, eng.alpha, eng.beta, eng.logp] = hmmFwdBack(eng.pi, eng.A, eng.B);
                       end
                        [eng.xi_summed,eng.xi] = hmmComputeTwoSlice(eng.alpha, eng.beta, eng.A, eng.B);
                    end
                    m = eng.xi(:,:,onto(1));
                    return;
                end
                %% Arbitrary marginal requested - use varelem
                dgm = convertToDgm(eng.model,horizon);
                visVals = eng.model.visVals;
                n = numel(visVals);
                if n > 0
                   dgm = condition(dgm,horizon+1:horizon+n,visVals);
                end
                m = marginal(dgm,onto);
            end
            
            %}
        
      
        function [logp,eng] = logprob(eng)
            checkParams(eng);
            if(isempty(eng.logp))
                [eng.logp] = hmmFwd(eng.pi, eng.A, eng.B);
                logp = eng.logp;
            end
            
        end
        
        function [s,eng] = sample(eng,nsamples)
            checkParams(eng);
            s = hmmSamplePost(eng.pi, eng.A, eng.B, nsamples);           
        end
                 
        function eng =  reset(eng)
        % Clears stored values used in lazy evaluation. This is called
        % automatically when the parameters are changed. 
          eng.alpha         = [];  
          eng.beta          = [];    
          eng.gamma         = [];   
          eng.xi            = [];      
          eng.xi_summed     = [];
          eng.viterbiPath   = []; 
          eng.logp          = [];  
          
        end
        
        function eng = precompute(eng)
        % Precompute values rather than rely on lazy evaluation.     
            [eng.gamma, eng.alpha, eng.beta, eng.logp] = hmmFwdBack(eng.pi, eng.A, eng.B);
             [eng.xi_summed,eng.xi] = hmmComputeTwoSlice(eng.alpha, eng.beta, eng.A, eng.B);
           
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

