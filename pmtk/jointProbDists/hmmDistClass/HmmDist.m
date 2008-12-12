classdef HmmDist < ParamDist
% This class represents a Hidden Markov Model. 
%
    properties                           
        nstates;                    % number of hidden states

        pi;                         % initial/starting distribution over hidden states 

        transitionMatrix;           % matrix of size nstates-by-nstates
                                    % transitionMatrix(i,j) = p( S(t) = j | S(t-1) = i), 
                                    
        observationModel            % The observation model to use - an empty 
                                    % object, e.g. MvnDist() or DiscreteDist()
                                    % that will be used as a template in
                                    % constructing all of the state conditional
                                    % densities. Alternatively, you can specify
                                    % the pre-initialized densities directly by
                                    % passing these into
                                    % stateConditionalDensities, below. 
       
        stateConditionalDensities;  % the observation model - one object per 
                                    % hidden state stored as a cell array. 
                                    % Each state conditional density must
                                    % support fit() via sufficient statistics 
                                    % with the name, 'suffStat', i.e.
                                    % fit(obj,'suffStat',SS) as well as
                                    % mkSuffStat(obj,X,weights), which
                                    % computes weighted (expected) sufficient
                                    % statistics in a format recognized by
                                    % fit(). Tied parameters, if any, are
                                    % represented as SharedParam objects, (i.e.
                                    % pointers to a shared data source).
        verbose = true;             
        
    end
    
    properties(GetAccess = 'private', SetAccess = 'private')
        obsDims;   % the dimensionality of an observation at a single time point t. 
    end
    
    methods
        
        function model = HmmDist(varargin)
        % Construct a new HMM with the specified number of hidden states. 
        %
        % FORMAT: 
        %            model = HmmDist('name1',val1,'name2',val2,...)
        %
        % INPUT:   
        %           'nstates'           - the number of hidden states
        
        %
        % - optional
        % 
        %           'pi'                           - the distribution initial hidden states
        %           'transitionMatrix'             - the transition matrix
        %           'stateConditionalDensities'    - see property above
        %           'ndimensions'                  - the dimensionality of an observation
        %                                            at a single time step
        %           'verbose'   
        %                                            
        %
        % OUTPUT:   'model'   - the constructed HMM object
        %
            [model.nstates,model.pi,model.transitionMatrix,model.stateConditionalDensities,model.obsDims,model.verbose,model.observationModel]...
                = process_options(varargin,'nstates',[],'pi',[],'transitionMatrix',[],'stateConditionalDensities',{},'ndimensions',1,'verbose',true,'observationModel',[]);
            
            if(isempty(model.nstates))
                model.nstates = numel(model.stateConditionalDensities);
            end
           
        end
        
        
        function model = fit(model,varargin)
        % Learn the parameters of the HMM from data.  
        %
        % FORMAT: 
        %           model = fit(model,'name1',val1,'name2',val2,...)
        %
        % INPUT:   
        %
        % 'data'             - a set of observation sequences - let 'n' be the
        %                      number of observed sequences, and 'd' be the
        %                      dimensionality of these sequences. If they are
        %                      all of the same length, 't', then data is of size
        %                      d-by-t-by-n, otherwise, data is a cell array such
        %                      that data{ex}{i,j} is the ith dimension at time
        %                      step j in example ex.
        %
        %
        % 'stateConditionalDensities - a cell array of the state conditional
        %                              densities - one per state, initialized to 
        %                              a starting guess. 
        %
        % 'pi0'                      - an initialization for the pi parameter, if
        %                              not specified, a random starting point is chosen. 
        %
        % 'transitionMatrix0'        - an initialization for the transition
        %                              matrix; if not specified, a random stochastic 
        %                              matrix is used instead. 
        % 
        %
        % 'latentValues'     - optional values for the latent variables in the 
        %                      case of fully observable data.
        %                     
        %
        % 'method'           - ['map'] | 'mle' | 'bayesian'
        %
        % 'algorithm'        -  ['em']  the fitting algorithm to use
        %
        % 'piPrior'          - a DirichletDist object
        %
        % 'transitionPrior'  - either a single DirichletDist object used as a
        %                      prior for each row of the transition matrix or a
        %                      cell array of DirichletDist objects, one for each
        %                      row. 
        %
        % 'observationPrior' - a single object acting as the prior for each
        %                      stateConditionalDensity. This must be a supported
        %                      prior distribution, i.e. the fit method of the
        %                      state conditional density must know what to do
        %                      with it in the call
        %                      fit(obj,'prior',observationPrior);
        %                      
        %
        % Any additional arguments are passed directly to the implementation of the
        % fit algorithm.
        %
        % If model.stateConditionalDensities is non-empty, these objects are
        % used to initialize the fitting algorithm. Similarly for model.pi and
        % model.transitionMatrix.
            
             [data,latentValues,method,algorithm,stateConditionalDensities,...
              model.pi,model.transitionMatrix,piPrior,transitionPrior,observationPrior,options]...
              = process_options(varargin,...
                 'data'             ,[]         ,...
                 'latentValues'     ,[]         ,...
                 'method'           ,'map'      ,...
                 'algorithm'        ,'em'       ,...
                 'stateConditionalDensities' ,{},...
                 'pi0'              ,[]         ,...
                 'transitionMatrix0',[]         ,...
                 'piPrior'          ,[]         ,...
                 'transitionPrior'  ,[]         ,...
                 'observationPrior' ,[]         );
             
             
             if(~isempty(latentValues)),error('fully observable data case not yet implemented');end
             if(~isempty(stateConditionalDensities))
                 model.stateConditionalDensities = stateConditionalDensities;
             end
             data = checkData(model,data);
             
             
             switch lower(algorithm)
                 case 'em'
                     model = emUpdate(model,data,piPrior,transitionPrior,observationPrior,options{:});
                 otherwise
                     error('%s is not a valid mle/map algorithm',algorithm);
             end
             
        end

        function logp = logprob(model,X)
        % logp(i) = log p(X{i} | model)
            n = nobservations(model,X);                              
            logp = zeros(n,1);
            for i=1:n
                logp(i) = logprob(predict(model,getObservation(model,X,i)));
            end
        end
        
        
        function [observed,hidden] = sample(model,nsamples,length)
            hidden = mc_sample(model.pi,model.transitionMatrix,length,nsamples);
            observed = zeros(model.ndimensions,length,nsamples);
            for n=1:nsamples
               for t=1:length
                   observed(:,t,n) = rowvec(sample(model.stateConditionalDensities{hidden(n,t)}));
               end
            end    
        end
        
        function trellis = predict(model,observation)
        % not yet vectorized, call with a single observation  
            n = nobservations(model,observation);  
            if(n > 1), error('Sorry, predict is not yet vectorized - please pass in each observation one at a time in a for loop. In future versions, calling predict with multiple observations will return a TrellisProductDist.');end
            trellis = TrellisDist(model.pi,model.transitionMatrix,makeLocalEvidence(model,observation));
        end
            
        function d = ndimensions(model)
            d = model.obsDims;
        end
 
    end
    
    methods(Access = 'protected')
         
        function model = emUpdate(model,data,piPrior,transitionPrior,observationPrior,varargin)
        % Update the transition matrix, the state conditional densities and pi,
        % the distribution over starting hidden states, using em.
        
        %% INIT
               [optTol,maxIter,clampPi,clampObs,clampTrans,other] = ...
                   process_options(varargin ,...
                   'optTol'                ,1e-4   ,...
                   'maxIter'               ,100    ,...
                   'clampPi'               ,false  ,...
                   'clampObs'              ,false  ,...
                   'clampTrans'            ,false  );
           
               if(clampPi && clampObs && clampTrans),return;end % nothing to do
               loglikelihood = 0;       
               iter = 1;
               converged = false;
               nobs  = nobservations(model,data);  
               model = initializeParams(model,data);
               if(~clampPi)     ,essPi      = zeros(model.nstates,1)            ;end % The expected number of visits to state one - needed to update pi
               if(~clampTrans)  ,essTrans   = zeros(model.nstates,model.nstates);end % The expected number of transitions from S(i) to S(j) - needed to update transmatrix 
               
               [stackedData,seqndx] = HmmDist.stackObservations(data);
               if(~clampObs), weightingMatrix = zeros(size(stackedData,1),model.nstates);end
               
               while(iter <= maxIter && ~converged)
                   prevLL = loglikelihood;
                   loglikelihood = 0;
                   if(~clampPi)     ,essPi(:)           = 0;end
                   if(~clampTrans)  ,essTrans(:)        = 0;end
                   if(~clampObs)    ,weightingMatrix(:) = 0;end
                   %% E Step
                   for j=1:nobs
                       trellis = predict(model,getObservation(model,data,j));
                       if(~clampPi)     ,essPi    = essPi    +  colvec(marginal(trellis,1));end  % marginal(trellis,1) is one slice marginal at t=1
                       if(~clampTrans)  ,essTrans = essTrans +  marginal(trellis)          ;end  % marginal(trellis) = two slice marginal xi
                       if(~clampObs)
                           gamma = marginal(trellis,':');
                           weightingMatrix(seqndx(j):seqndx(j)+size(gamma,2)-1,:) =...
                             weightingMatrix(seqndx(j):seqndx(j)+size(gamma,2)-1,:) + gamma';
                       end
                       loglikelihood = loglikelihood + logprob(trellis);
                   end
                   if(~clampObs)
                       essObs = cell(model.nstates,1);                          % observation model expected sufficient statistics
                       for i=1:model.nstates
                           essObs{i} = mkSuffStat(model.stateConditionalDensities{i},stackedData,weightingMatrix(:,i)); 
                       end
                   end
                   %% M Step PI
                   if(~clampPi)
                       if(isempty(piPrior))
                            model.pi = normalize(essPi);
                       else
                            model.pi = mean(DirichletDist(essPi + colvec(piPrior.alpha)));
                       end
                   end
                   %% M Step Transition Matrix
                   if(~clampTrans)
                       if(isempty(transitionPrior))
                           model.transitionMatrix = normalize(essTrans,2);
                       else
                           if(numel(transitionPrior) == 1)
                               if(iscell(transitionPrior))
                                   transitionPrior = transitionPrior{:};
                               end
                               model.transitionMatrix = normalize(bsxfun(@plus,essTrans,rowvec(transitionPrior.alpha)),2);
                           else
                               for i=1:size(model.transitionMatrix,1)
                                  model.transitionMatrix(i,:) = rowvec(mean(DirichletDist(essTrans(i,:) + transitionPrior{i}.alpha))); 
                               end
                           end
                       end
                   end
                   %% M Step Observation Model
                   if(~clampObs)
                       if(isTied(model.stateConditionalDensities{i})) % update the shared parameters first and then clamp them before updating the rest
                           % since the state conditional densitity will know if
                           % its tied or not, it can return appropriate suff
                           % stats.
                           model.stateConditionalDensities{i} = fit(model.stateConditionalDensities{i},'suffStat',essObs{i},'prior',observationPrior);
                           for i=2:model.nstates
                               model.stateConditionalDensities{i} = unclampTied(fit(clampTied(model.stateConditionalDensities{i}),'suffStat',essObs{i},'prior',observationPrior));
                           end
                       else
                           for i=1:model.nstates
                               model.stateConditionalDensities{i} = fit(model.stateConditionalDensities{i},'suffStat',essObs{i},'prior',observationPrior);
                           end
                       end
                   end
                   %% Test Convergence
                   if(model.verbose)
                      fprintf('\niteration %d, loglik = %f\n',iter,loglikelihood); 
                   end
                   iter = iter + 1;
                   converged = (abs(loglikelihood - prevLL) / (abs(loglikelihood) + abs(prevLL) + eps)/2) < optTol;
               end % end of em loop
        end % end of emUpdate method
 
        function model = initializeParams(model,X)                                        
        % Initialize parameters to starting states in preperation for EM.
            if(isempty(model.transitionMatrix))
               model.transitionMatrix = normalize(rand(model.nstates),2); 
            end
            if(isempty(model.pi))
               model.pi = normalize(ones(1,model.nstates)); 
            end
            if(~isempty(model.stateConditionalDensities))
                if(isempty(model.observationModel))
                    model.observationModel = feval(class(model.stateConditionalDensities{1}));
                end
                return;
            end
            
            if(isempty(model.observationModel))
               error('Please specify an observation model or the state conditional densities directly'); 
            end
            if(isempty(model.nstates) || model.nstates == 0)
                error('Please specify the number of hidden states');
            end
            
            data = HmmDist.stackObservations(X);
            template = fit(model.observationModel,'data',data);
            model.stateConditionalDensities = copy(template,model.nstates,1);
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
           model.obsDims = d;
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
           [junk,n] = getObservation(model,X,1); %#ok
        end
        
        function localEvidence = makeLocalEvidence(model,obs)
        % the probability of the observed sequence under each state conditional density. 
        % localEvidence(i,t) = p(y(t) | S(t)=i)
            localEvidence = zeros(model.nstates,size(obs,2));     
            for i = 1:model.nstates
                localEvidence(i,:) = exp(logprob(model.stateConditionalDensities{i},obs'));
            end
            
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
            trueObsModel = {DiscreteDist(ones(1,6)./6);DiscreteDist([ones(1,5)./10,0.5])};
            trueTransmat = [0.8,0.2;0.1,0.90];
            truePi = [0.5,0.5];
            truth = HmmDist('pi',truePi,'transitionMatrix',trueTransmat,'stateConditionalDensities',trueObsModel);
            nsamples = 200; length1 = 13; length2 = 30;
            [observed1,hidden1] = sample(truth,nsamples/2,length1);
            [observed2,hidden2] = sample(truth,nsamples/2,length2);
            observed = [num2cell(squeeze(observed1),1)';num2cell(squeeze(observed2),1)'];
            
            model = HmmDist('observationModel',DiscreteDist(),'nstates',2);
            model = fit(model,'data',observed);
            trellis = predict(model,observed{1}');
            
            postSample = mode(sample(trellis,1000),2)' %#ok
            viterbi  = mode(trellis)                   %#ok
            maxmarg = maxidx(marginal(trellis,':'))    %#ok
            %% MVN
            trueObsModel = {MvnDist(zeros(1,10),randpd(10));MvnDist(ones(1,10),randpd(10))};
            trueTransmat  = [0.8,0.2;0.3,0.7];
            truePi       = [0.5,0.5];
            trueModel = HmmDist('pi',truePi,'transitionMatrix',trueTransmat,'stateConditionalDensities',trueObsModel,'ndimensions',10);
            nsamples = 200; length = 20;
            [observed,trueHidden] = sample(trueModel,nsamples,length);

            
            model = HmmDist('observationModel',MvnDist(),'nstates',2);
            model = fit(model,'data',observed,'observationPrior',InvWishartDist(10,diag(0.1*ones(1,10))),'transitionPrior',DirichletDist([0.1,0.1]),'PiPrior',DirichletDist([0.1,0.1]));
            
            
        end
      
        
        function seqalign()
            setSeed(10);
            load data45; nstates = 5; 
            pi0 = [1,0,0,0,0];
            transmat0 = normalize(diag(ones(nstates,1)) + diag(ones(nstates-1,1),1),2);
            condDensity = HmmDist('nstates',5,'observationModel',MvnDist());
            model = GenerativeClassifierDist('classConditionals',condDensity,'nclasses',2,'classSupport',4:5);
            obsData = {train4,train5}; 
            hidData = [4,5];
            fitOptions = {'transitionMatrix0',transmat0,'pi0',pi0,'maxIter',5};
            
            model = fit(model,'dataObs',obsData,'dataHid',hidData,'fitOptions',fitOptions);
            pred = predict(model,test45);
            yhat = mode(pred);
            nerrors = sum(yhat ~= labels');
            display(nerrors);
            
            if(exist('specgram','file'))
                model4  = model.classConditionalDensities{1};
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

