classdef HmmDist 
% This class represents a Hidden Markov Model. 
    properties                           
        nstates;                    % number of hidden states

        startDist;                  % distribution over starting states: 
                                    % a DiscreteDist or Discrete_DirichletDist

        transitionDist;             % p( S(t) = j | S(t-1) = i),
                                    % A DiscreteDist or a Discrete_DirichletDist
                                    % (vectorized) (Each *column* of
                                    % mean(transitionDist) sums to one)
                                     
        emissionDist;               % the emmission densities - one object per 
                                    % hidden state stored as a cell array. 
                                    % Each state conditional density must
                                    % support fit() via sufficient statistics 
                                    % with the name, 'suffStat', i.e.
                                    % fit(obj,'suffStat',SS) as well as
                                    % obj.mkSuffStat(X,weights), which
                                    % computes weighted (expected) sufficient
                                    % statistics in a format recognized by
                                    % fit(). If only one object is specified, it
                                    % is automatically initialized and
                                    % replicated. 
                                    
     
                                    
        verbose = true;   
     
      
    end
    
    properties(GetAccess = 'protected', SetAccess = 'protected')
       initWithData = false; 
    end
    
    methods
        
        function model = HmmDist(varargin)
        % Construct a new HMM with the specified number of hidden states. 
        % FORMAT:    model = HmmDist('name1',val1,'name2',val2,...)
        % INPUT:    'nstates'                       - the number of hidden states
        %           'transitionDist'                - see property above   
        %           'emissionDist'                  - see proeprty above
        %           'infEng'                        - an inference engine for
        %                                             answering condition and 
        %                                             marginal queries, etc - one
        %                                             is initialized by default. 
        %           'verbose'                       - see property above
        % OUTPUT:   'model'   - the constructed HMM object
            if nargin == 0; return; end
            [model.nstates                          ,...
             model.startDist                        ,...
             model.transitionDist                   ,...
             model.emissionDist                     ,...
             model.verbose                          ]...
             = process_options(varargin,...
                'nstates'                           ,[],...
                'startDist'                         ,[],...
                'transitionDist'                    ,[],...
                'emissionDist'                      ,{},...
                'verbose'                           ,true);
            if(isempty(model.nstates))
                model.nstates = numel(model.emissionDist);
            end
            if(isempty(model.transitionDist))
               model.transitionDist = DiscreteDist('mu',normalize(rand(model.nstates),1),'support',1:model.nstates); 
            end
            if(isempty(model.startDist))
               model.startDist = DiscreteDist('mu',normalize(rand(model.nstates,1)),'support',1:model.nstates); 
            end
            if(numel(model.emissionDist) == 1)
                model.emissionDist = copy(model.emissionDist,model.nstates,1);
                model.initWithData = true;
            end
           
            
            model.domain = struct;
            model.domain.Z = 1:model.nstates;
            model.domain.Y = [];
            if(~isempty(model.emissionDist))
                if(iscell(model.emissionDist))
                    model.domain.Y = 1:ndimensions(model.emissionDist{1});
                else
                    model.domain.Y = 1:ndimensions(model.emissionDist);
                end
                
            end
        end
        
        function dgm = convertToDgm(model,t)
          % Return a DgmDist() that is equivalent to this HMM up to time step t -
          % that is a truncated version of this HMM.
          CPD{1} = TabularCPD(mean(model.startDist)');
          CPD(2:t) = copy(TabularCPD(mean(model.transitionDist)'),1,t-1);
          CPD(t+1:2*t) = copy(MixtureDist('distributions',model.emissionDist),1,t);

          G = zeros(2*t);
          for i=1:t-1
            G(i,i+1) = 1;
            G(i,i+t) = 1;
          end
          G(t,2*t) = 1;
          dgm = DgmDist(G,'CPDs',CPD,'domain',1:2*t,'InfEng',VarElimInfEng());
        end

        
        
        function model = fit(model,varargin)
          % Learn the parameters of the HMM from data.
          %
          % FORMAT:   model = fit(model,'name1',val1,'name2',val2,...)
          %
          % INPUT:   'data'   i=obs, t=time, ex=sequence
          % If all sequences are same length, use data(i,t,ex) 
          % If of differnet length, use data{ex}(i,t) 
          [data,options] = process_options(varargin,'data',[]);
          model = hmmFitEm(model,data,options{:});
        end
        
        function logp = logprob(model,Y)
          % logp = log p(Y|model) where Y is a d*T vector of *visible* values
          % For multiple sequences of different lengths, Y{i} is d*T
          % For multiple sequences of same length, Y is d*T*n
          n = nobservations(model,Y);
          logp = zeros(n,1);
          A = model.transitionDist;
          pi = model.startDist;
          for i=1:n
            seq = getObservation(model, Y, i);
            B =  makeLocalEvidence(model,seq);
            [logp(i)] = hmmFwd(pi, A, B);
          end
        end
        
        function [observed,hidden] = sample(model,nsamples,length)
          % Sample nsamples all of the specified length from this HMM.
          hidden = mc_sample(mean(model.startDist),mean(model.transitionDist)',length,nsamples);
          observed = zeros(model.ndimensions,length,nsamples);
          for n=1:nsamples
            for t=1:length
              observed(:,t,n) = rowvec(sample(model.emissionDist{hidden(n,t)}));
            end
          end
        end
        
        function S = samplePost(model,nsamples)
        % Forwards filtering, backwards sampling    
            if(~model.conditioned)
                error('You must first call condition.'); 
            end
            S = sample(model.infEng,nsamples);
        end
        
        function d = ndimensions(model)
        % The dimensionality of the emission densities.    
           d = ndimensions(model.emissionDist{1});
        end
        
         function localEvidence = makeLocalEvidence(model,obs)
         % the probability of the observed sequence under each state conditional density. 
         % localEvidence(i,t) = p(Y(t) | Z(t)=i)
            if nargin < 2 || isempty(obs)
               localEvidence = ones(model.nstates,1); % probability of an empty event is 1 - size of matrix will be expanded when needed
               return;
            end
            localEvidence = zeros(model.nstates,size(obs,2));
            for i = 1:model.nstates
                localEvidence(i,:) = exp(logprob(model.emissionDist{i},obs'));
            end
         end
        
         function [obs,n] = getObservation(model,X,i)                            %#ok
           % Get the ith observation/example from X.
           switch class(X)
             case 'cell'
               n = numel(X);
               obs = X{i};
             case 'double'
               n = size(X,3);
               obs = X(:,:,i);
           end
         end
        
        function n = nobservations(model,X)
        % determine how many observations are in X    
           [junk,n] = getObservation(model,X,1); %#ok
        end
 
    end
    

  
end % end of class

