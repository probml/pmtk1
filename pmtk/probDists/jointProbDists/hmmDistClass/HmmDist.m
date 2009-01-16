classdef HmmDist < ParamJointDist
% This class represents a Hidden Markov Model. 
    properties                           
        nstates;                    % number of hidden states

        startDist;                  % distribution over starting states: 
                                    % a DiscreteDist or Discrete_DirichletDist

        transitionDist;             % p( S(t) = j | S(t-1) = i),
                                    % A DiscreteDist or a Discrete_DirichletDist
                                    % (vectorized)
                                     
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
            [model.nstates                          ,...
             model.startDist                        ,...
             model.transitionDist                   ,...
             model.emissionDist                     ,...
             model.infEng                           ,...
             model.verbose                          ]...
             = process_options(varargin,...
                'nstates'                           ,[],...
                'startDist'                         ,[],...
                'transitionDist'                    ,[],...
                'emissionDist'                      ,{},...
                'infEng'                            ,FwdBackInfEng(),...
                'verbose'                           ,true);
            if(isempty(model.nstates))
                model.nstates = numel(model.emissionDist);
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
        
        function model = fit(model,varargin)
        % Learn the parameters of the HMM from data.  
        %
        % FORMAT:   model = fit(model,'name1',val1,'name2',val2,...)
        %
        % INPUT:   'data'    - a set of observation sequences - let 'n' be the
        %                      number of observed sequences, and 'd' be the
        %                      dimensionality of these sequences. If they are
        %                      all of the same length, 't', then data is of size
        %                      d-by-t-by-n, otherwise, data is a cell array such
        %                      that data{ex}{i,j} is the ith dimension at time
        %                      step j in example ex.
        
             [data,options] = process_options(varargin,'data',[]);
             data = checkData(model,data);
             model = emUpdate(model,data,options{:});
        end
        
        function logp = logprob(model,Y)
        % logp(i) = log p(Y{i} | model)
        % If Y specified, the model is first conditioned on Y. Y may store
        % multiple observations, in which case the model is conditioned on each
        % in turn. 
            if(nargin ==1)
                if(~model.conditioned)
                   error('You must call condition first or specify an observation'); 
                end
                [logp,model.infEng] = logprob(model.infEng);
            else
                n = nobservations(model,Y);
                logp = zeros(n,1);
                for i=1:n
                    model = condition(model,'Y',getObservation(model,Y,i));
                    [logp(i),model.infEng] = logprob(model.infEng);
                end
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
            S = sample(model.infEng,nsamples);
        end
        
        function d = ndimensions(model)
        % The dimensionality of the emission densities.    
           d = ndimensions(model.emissionDist{1});
        end
        
         function localEvidence = makeLocalEvidence(model,obs)
         % the probability of the observed sequence under each state conditional density. 
         % localEvidence(i,t) = p(Y(t) | Z(t)=i)
            localEvidence = zeros(model.nstates,size(obs,2));     
            for i = 1:model.nstates
                localEvidence(i,:) = exp(logprob(model.emissionDist{i},obs'));
            end
        end
 
    end
    
    methods(Access = 'protected')
         
        function model = emUpdate(model,data,varargin)
        % Update all of the parameters of the HMM via EM
        % We use the abriviation ess for expected sufficient statistics
               [optTol,maxIter,clampedStart,clampedObs,clampedTrans] = ...
                   process_options(varargin ,...
                   'optTol'                ,1e-4   ,...
                   'maxIter'               ,100    ,...
                   'clampedStart'          ,false  ,...
                   'clampedObs'            ,false  ,...
                   'clampedTrans'          ,false  );
               if(clampedStart && clampedObs && clampedTrans),return;end % nothing to do
               
               [loglikelihood,prevLL,iter,converged,nobs,essStart,...
                essTrans,essObs,stackedData,seqndx,weightingMatrix] = initializeVariables();
               % these variables are global to the emUpdate method
               
               %% EM LOOP
               while(iter <= maxIter && not(converged))
                   resetStatistics();
                   Estep();
                   Mstep();
                   testConvergence();   
               end 
                             
            function Estep()
            % Compute the expected sufficient statistics    
                for j=1:nobs
                    model = condition(model,'Y',getObservation(model,data,j));
                    %% Starting Distribution
                    if(not(clampedStart)) 
                        essStart.counts = essStart.counts + colvec(marginal(model,1));     % marginal(model,1) is one slice marginal at t=1
                    end 
                    %% Transition Distributions
                    if(not(clampedTrans))
                        essTrans.counts = essTrans.counts +  marginal(model);               % marginal(model) = full two slice marginals xi
                    end  
                    if(not(clampedObs))
                        gamma = marginal(model,':');                                        % marginal(model,':') all of the 1 slice marginals, i.e. gamma
                        weightingMatrix(seqndx(j):seqndx(j)+size(gamma,2)-1,:) =...
                          weightingMatrix(seqndx(j):seqndx(j)+size(gamma,2)-1,:) + gamma';
                    end
                    loglikelihood = loglikelihood + logprob(model);
                end
                essTrans.counts = essTrans.counts';
                %% Emission Distributions
                if(not(clampedObs))
                    for i=1:model.nstates
                        essObs{i} = model.emissionDist{i}.mkSuffStat(stackedData,weightingMatrix(:,i));
                    end
                end
            end % end of Estep subfunction
            
            function Mstep()
            % Maximize with respect to the ess calculated in the previous Estep    
                %% Starting Distribution
                if(not(clampedStart))
                    model.startDist = fit(model.startDist,'suffStat',essStart);
                end
                %% M Step Transition Matrix
                if(not(clampedTrans))
                     model.transitionDist = fit(model.transitionDist,'suffStat',essTrans);
                end
                %% M Step Observation Model
                if(not(clampedObs))
                    if(isTied(model.emissionDist{1})) % update the shared parameters first and then clamp them before updating the rest
                        % since the state conditional densitity will know if
                        % its tied or not, it can return appropriate suffStats.
                        model.emissionDist{1} = fit(model.emissionDist{1},'suffStat',essObs{1});
                        for i=2:model.nstates
                            model.emissionDist{i} = unclampTied(fit(clampTied(model.emissionDist{i}),'suffStat',essObs{i}));
                        end
                    else
                        for i=1:model.nstates
                            model.emissionDist{i} = fit(model.emissionDist{i},'suffStat',essObs{i});
                        end
                    end
                end
            end % end of Mstep subfunction
            
            function testConvergence()
            % Test if EM has converged yet  
                if(model.verbose)
                    fprintf('\niteration %d, loglik = %f\n',iter,loglikelihood);
                end
                iter = iter + 1;
                converged = (abs(loglikelihood - prevLL) / (abs(loglikelihood) + abs(prevLL) + eps)/2) < optTol;
            end % end of testConvergence subfunction
            
            function resetStatistics()
            % called during EM loop to reset stats    
                prevLL = loglikelihood;
                loglikelihood = 0;
                if(~clampedStart)  ,essStart.counts(:) = 0;end
                if(~clampedTrans)  ,essTrans.counts(:) = 0;end
                if(~clampedObs)    ,weightingMatrix(:) = 0;end
            end % end of resetStatistics subfunction
            
            function [loglikelihood,prevLL,iter,converged,nobs,essStart,essTrans,essObs,stackedData,seqndx,weightingMatrix] = initializeVariables()
            % called prior to EM loop to setup variables   
                loglikelihood = 0;
                prevLL = 0;
                iter = 1;
                converged = false;
                nobs  = nobservations(model,data);
                model = initializeParams(model,data);
                
                if(~clampedStart) ,essStart.counts = zeros(model.nstates,1)            ;end % The expected number of visits to state one - needed to update startDist
                if(~clampedTrans) ,essTrans.counts = zeros(model.nstates,model.nstates);end % The expected number of transitions from S(i) to S(j) - needed to update transDist
                if(~clampedObs)   , essObs = cell(model.nstates,1);end
                [stackedData,seqndx] = HmmDist.stackObservations(data);
                if(~clampedObs), weightingMatrix = zeros(size(stackedData,1),model.nstates);end
            end % end of initializeStatistics subfunction
 
        end % end of emUpdate method
 
        
 
        function model = initializeParams(model,X)                                        
        % Initialize parameters to starting states in preperation for EM.
            if(isempty(model.transitionDist))
               model.transitionDist = DiscreteDist('mu',normalize(rand(model.nstates),1),'support',1:model.nstates); 
            end
            if(isempty(model.startDist))
               model.startDist = DiscreteDist('mu',normalize(ones(model.nstates,1)),'support',1:model.nstates); 
            end
            if(numel(model.emissionDist) == 1)
                 if(~iscell(model.emissionDist))
                     model.emissionDist = {model.emissionDist};
                 end
                 data = HmmDist.stackObservations(X);
                 template = fit(model.emissionDist{1},'data',data);
                 model.emissionDist = copy(template,model.nstates,1);
            end
        end
        
         function data = checkData(model,data)
         % basic checks to make sure the data is in the right format
           if(isempty(data))
               error('You must specify data to fit this object');
           end
           switch class(data)
               case 'cell'
                   if(numel(data) == 1 && iscell(data{1}))
                       data = data{:};
                   end
                   data = rowvec(data);
                   n = numel(data);
                   d = size(data{1},1);
                   transpose = false;
                   for i=2:n
                      if(size(data{i},1) ~= d)
                          transpose = true;
                          break;
                      end
                   end
                   if(transpose)
                      d = size(data{1},2);
                      data{1} = data{1}';
                      for i=2:n
                        data{i} = data{i}';
                        if(size(data{i},1) ~= d)
                           error('Observations must be of the same dimensionality.');
                        end
                      end
                   end
                   if(model.verbose)
                       fprintf('\nInterpreting data as %d observation sequences,\nwhere each sequence is comprised of a variable\nnumber of %d-dimensional observations.\n',n,d);
                   end
               case 'double'
                   if(model.verbose)
                       [d,t,n] = size(data);
                       fprintf('\nInterpreting data as %d observation sequences,\nwhere each sequence is comprised of %d\n%d-dimensional observations.\n',n,t,d);
                   end    
               otherwise
                   error('Data must be either a matrix of double values or a cell array');
           end
        end % end of checkData method
        
        
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
        
       
        
    end % end of protected methods
    
    methods(Static = true)
        
        function [X,ndx] = stackObservations(data)
        % data is a cell array of sequences of different length but with the
        % same dimensionality. X is a matrix of all of these sequences stacked
        % together in an n-by-d matrix where n is the sum of the lengths of all
        % of the sequences and d is the shared dimensionality. Within each cell
        % of data, the first dimension is d and the second is the length of the
        % observation. ndx stores the indices into X corresponding to the start
        % of each new sequence. 
        %
        % Alternatively, if data is a 3d matrix of size d-t-n, data is simply
        % reshaped into size []-d and ndx is evenly spaced.
            
            if(iscell(data))
                X = cell2mat(data)';
                ndx = cumsum([1,cell2mat(cellfun(@(seq)size(seq,2),data,'UniformOutput',false))]);
                ndx = ndx(1:end-1);
            else
                X = reshape(data,[],size(data,1));
                ndx = cumsum([1,size(data,2)*ones(1,size(data,3))]);
                ndx = ndx(1:end-1);
            end
        end
    end
    
    methods(Static = true)
        
        function testClass()
            setSeed(0);
            trueObsModel = {DiscreteDist('mu',ones(6,1)./6       ,'support',1:6)
                            DiscreteDist('mu',[ones(5,1)./10;0.5],'support',1:6)};
           
            trueTransDist = DiscreteDist('mu',[0.8,0.2;0.3,0.70]','support',1:2);
            trueStartDist = DiscreteDist('mu',[0.5,0.5],'support',1:2);
            trueModel = HmmDist('startDist'     ,trueStartDist,...
                                'transitionDist',trueTransDist,...
                                'emissionDist'  ,trueObsModel);
            
            nsamples = 200; length1 = 13; length2 = 30;
            [observed1,hidden1] = sample(trueModel,nsamples/2,length1);
            [observed2,hidden2] = sample(trueModel,nsamples/2,length2);
            observed = [num2cell(squeeze(observed1),1)';num2cell(squeeze(observed2),1)'];
            
            model = HmmDist('emissionDist',DiscreteDist('support',1:6),'nstates',2);
            model = fit(model,'data',observed);
            
            model = condition(model,'Y',observed{1}');
            postSample = mode(samplePost(model,1000),2)'
            viterbi = mode(model)
            maxmarg = maxidx(marginal(model,':'))
            %% MVN
            trueObsModel = {MvnDist(zeros(1,10),randpd(10));MvnDist(ones(1,10),randpd(10))};
            trueTransDist = DiscreteDist('mu',[0.8,0.2;0.1,0.90]','support',1:2);
            trueStartDist = DiscreteDist('mu',[0.5,0.5],'support',1:2);
            trueModel = HmmDist('startDist'     ,trueStartDist,...
                                'transitionDist',trueTransDist,...
                                'emissionDist'  ,trueObsModel);
            nsamples = 200; length = 20;
            [observed,trueHidden] = sample(trueModel,nsamples,length);

            
            model = HmmDist('emissionDist',MvnDist(),'nstates',2);
            model = fit(model,'data',observed);
        end
      
        
        function seqalign()
            if(~exist('data45.mat','file'))
               error('Please download data45.mat from www.cs.ubc.ca/~murphyk/pmtk and save it in the data directory');
            end
            setSeed(0);
            load data45; nstates = 5; ndimensions = 13;
            startDist = DiscreteDist('mu',[1,0,0,0,0]','support',1:5);
            transmat0 = normalize(diag(ones(nstates,1)) + diag(ones(nstates-1,1),1),2);
            transDist = DiscreteDist('mu',transmat0','support',1:5);
            emissionDist = cell(5,1);
            for i=1:nstates
                emissionDist{i} = mkRndParams(MvnDist(),ndimensions);
            end
            model4 = HmmDist('startDist',startDist,'transitionDist',transDist,'emissionDist',emissionDist,'nstates',nstates);
            model4 = fit(model4,'data',train4);
            if(exist('specgram','file'))
                subplot(2,2,1);
                specgram(signal1); 
                subplot(2,2,2)
                specgram(signal2);
                subplot(2,2,3);
                plot(mode(predict(model4,mfcc1)));
                set(gca,'YTick',1:5);
                subplot(2,2,4);
                plot(mode(predict(model4,mfcc2)));
                set(gca,'YTick',1:5);
                maximizeFigure;
            end 
        end
      
        
    end
    
    
    
    
    
    
end % end of class

