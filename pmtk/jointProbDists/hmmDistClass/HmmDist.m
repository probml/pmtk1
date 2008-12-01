classdef HmmDist < ParamDist
% This class represents a Hidden Markov Model. 
%
% NOTATION: y - observed data, y(t) is the t-th observation
%           S - hidden state , S(t) is the hidden state corresponding to the
%                                   t-th observation.
    
    properties                           
        nstates;                    % number of hidden states

        pi;                         % initial/starting distribution over hidden states 

        transitionMatrix;           % matrix of size nstates-by-nstates
                                    % transitionMatrix(i,j) = p( S(t) = j | S(t-1) = i), 
                                    
                                    
        observationModel;           % The PMTK class name of the distributions 
                                    % that make up the observation model, i.e. 
                                    % 'DiscreteDist', 'MvnDist', etc. 
       
        stateConditionalDensities;  % the observation model - one object per 
                                    % hidden state stored as a cell array. 
                                    % Each state conditional density must
                                    % support fit() via sufficient statistics 
                                    % with the name, 'suffStat', i.e.
                                    % fit(obj,'suffStat',SS) as well as
                                    % makeSuffStat(obj,X,weights), which
                                    % computes weighted (expected) sufficient
                                    % statistics in a format recognized by
                                    % fit(). Tied parameters, if any, are
                                    % represented as SharedParam objects, (i.e.
                                    % pointers to a shared data source).
        verbose = true;             
        
        
    end
    
    properties(GetAccess = 'private', SetAccess = 'private')
        obsDims;   % the dimensionality of an observation at a single time point t. 
        nsymbols;  % if observation model is discrete, this is the number of output symbols.
        
    end
    
    methods
        
        function model = HmmDist(varargin)
        % Construct a new HMM with the specified number of hidden states. 
        %
        % FORMAT: 
        %            model = HmmDist('name1',val1,'name2',val2,...)
        %
        % INPUT:   
        %           'nstates' - the number of hidden states
        %
        % OUTPUT:   'model'   - the constructed HMM object
        %
            model.nstates = process_options(varargin,'nstates',[]);
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
        % 'latentValues'     - optional values for the latent variables in the 
        %                      case of fully observable data.
        %                     
        %
        % 'observationModel' - the exact case sensitive name of the PMTK class
        %                      to be used for the observation model. For
        %                      instance, if you want each state conditional
        %                      density to be an MvnDist, this parameter must be
        %                      'MvnDist'. This can also be an object of the
        %                      class, e.g. MvnDist(). 
        %
        % 'method'           - ['map'] | 'mle' | 'bayesian'
        %
        % 'algorithm'        -  ['em']  the fitting algorithm to use
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
            
             [data,latentValues,method,algorithm,observationModel,transitionPrior,observationPrior,options] = process_options(varargin,...
                 'data'             ,[]         ,...
                 'latentValues'     ,[]         ,...
                 'method'           ,'map'      ,...
                 'algorithm'        ,'em'       ,...
                 'observationModel' ,''         ,...
                 'transitionPrior'  ,[]         ,...
                 'observationPrior' ,[]         );
             
             model.observationModel = observationModel;    
             if(~isempty(latentValues)),error('fully observable data case not yet implemented');end
             
             data = checkData(model,data);
             model = checkObservationModel(model,observationModel);
             
             switch lower(algorithm)
                 case 'em'
                     model = emUpdate(model,data,method,transitionPrior,observationPrior,options{:});
                 otherwise
                     error('%s is not a valid mle/map algorithm',algorithm);
             end
             
        end

        function logp = logprob(model,X)
         
            [junk,n] = model.getObservation(X,1);
            logp = zeros(n,1);
            for i=1:n
                [alpha,logp(i)] = hmmFilter(model.pi,model.transitionMatrix,makeLocalEvidence(model,getObservation(X,i)));
            end
        end
        
        
        function trellis = predict(model,observation)
        % not yet vectorized, call with a single observation  
            [junk,n] = getObservation(model,observation);
            if(n > 1), error('Sorry, predict is not yet vectorized - please pass in each observation one at a time in a for loop. In future versions, calling predict with multiple observations will return a TrellisProductDist.');end
            trellis = TrellisDist(model.pi,model.transitionMatrix,makeLocalEvidence(observation));
        end
            
        function d = ndimensions(model)
            d = model.obsDims;
        end
 
    end
    
    methods(Access = 'protected')
         
        function emUpdate(model,data,transitionPrior,observationPrior,varargin)
        % Update the transition matrix, the state conditional densities and pi,
        % the distribution over starting hidden states, using em.
        
        %% INIT
               [optTol,maxIter,clampPi,clampObs,clampTrans] = ...
                   process_options(varargin ,...
                   'optTol'                ,1e-4   ,...
                   'maxIter'               ,20     ,...
                   'clamp_pi'               ,false  ,...
                   'clamp_obs'              ,false  ,...
                   'clamp_trans'            ,false  );
           
                              
               loglikelihood = -inf;       
               iter = 1;
               converged = false;
               [junk,nobservations] = getObservation(model,data,1);
               
               essPi      = zeros(model.nstates,1);                    % The expected number of visits to state one - needed to update pi
               essTrans   = zeros(model.nstates,model.nstates);        % The expected number of transitions from S(i) to S(j) - needed to update transmatrix 
               [stackedData,seqndx] = HmmDist.stackData(data);
               weightingMatrix = zeros(size(stackedData,1),1);
               
               while(iter < maxIter && ~converged)
                   
                   prevLL = loglikelihood;
                   essPi(:) = 0; essTrans(:) = 0; weightingMatrix(:) = 0;
                   %% E Step
                   for j=1:nobservations
                       trellis = precompute(predict(model,getObservation(model,data,j)));
                       essPi = essPi + colvec(marginal(trellis,1));
                       essTrans = essTrans +  marginal(trellis);                 % marginal(trellis) = two slice marginal xi
                       
                       weight = colvec(sum(marginal(trellis,':'),2));            % marginal(trellis,':') = gamma
                       weightingMatrix(seqndx(j):seqndx(j)+size(weight,1)-1) = weight;
                       loglikelihood = loglikelihood + current_ll;
                   end
                   essObs = cell(model.nstates,1);                               % observation model expected sufficient statistics
                   for i=1:model.nstates
                       essObs{i} = makeSuffStat(model.stateConditionalDensities{i},stackedData,weightingMatrix);
                   end
                   
                   %% M Step
                   model.pi = normalize(essPi);    
                   if(isempty(transitionPrior))
                       model.transitionMatrix = normalize(essTrans,2);
                   else
                       if(numel(transitionPrior) == 1)
                           if(iscell(transitionPrior))
                               transitionPrior = transitionPrior{:};
                           end
                           switch class(transitionPrior)
                               case 'DirichletDist'
                                   model.transitionMatrix = normalize(bsxfun(@plus,essTrans,rowvec(transitionPrior.alpha)),2);
                               otherwise
                                   error('%s is an unsupported prior on the transition matrix.',class(transitionPrior));
                           end
                       elseif(numel(transitionPrior) == model.nstates)
                           alpha = zeros(size(model.transitionMatrix));
                           for i=1:numel(transitionPrior)
                               prior = transitionPrior{i};
                               if(~isa(prior,'DirichletDist'))
                                   error('%s is not a supported prior on a row of the transition matrix',class(prior));
                               end
                               alpha(i,:) = rowvec(prior.alpha);
                           end
                           model.transitionMatrix = normalize(essTrans + alpha - 1,2);
                       else
                           error('Inconsistent number of prior distributions on the transition matrix: must be 1 or nstates');
                       end
                   end
               
                   if(isempty(observationPrior))
                       for i=1:model.nstates
                           model.stateConditionalDensities{i} = fit(model.stateConditionalDensities{i},'suffStat',essObs{i});
                       end
                   else
                       for i=1:model.nstates
                           model.stateConditionalDensities{i} = fit(model.stateConditionalDensities{i},'suffStat',expSSobs,'prior',observationPrior);
                       end
                   end
                   
                   %% Test Convergence
                   iter = iter + 1;
                   converged = (abs(loglikelihood - prevLL) / (abs(loglikelihood) + abs(prev_ll) + eps)/2) < optTol;
               end % end of em loop
               
        end % end of emUpdate method
        
        
        
        
        function initializeParams(model,X)
        % Initialize parameters to starting states in preperation for EM.
           
            if(isempty(model.transitionMatrix))
               model.transitionMatrix = normalize(rand(model.nstates,model.nstates),2); 
            end
            if(isempty(model.pi))
               model.pi = normalize(ones(1,model.nstates)); 
            end
             
           % init state conditional densities here    
        end
        
        
        function [nsym,sym] = noutputSymbols(model,X)
        % Helper method to determine the number of output symbols in the case of 
        % a discrete observation model. 
            assert(strcmpi(model.observationModel,'DiscreteDist'));
            [junk,n] = getObervation(model,X,1);
            unq = [];
            for i=1:n
                unq = union(unq,unique(getObservation(model,X,i)));
            end
            nsym = numel(unq);
            sym = unq;
        end
        
        
         function data = checkData(model,data)
        % basic checks to make sure the data is in the right format
           if(isempty(data))
               error('You must specify data to fit this object');
           end
           
           switch class(data)
               case 'cell'
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
        
        
        function [obs,n] = getObservation(model,X,i)
        % Get the ith observation/example from X.     
            switch class(X)
                case 'cell'
                    n = numel(X);
                    obs = X{i};
                case 'double'
                    if(ndims(X) == 3)
                        n = size(X,3);
                        obs = X(:,:,i);
                    else
                        n = size(X,1);
                        obs = X(i,:);
                    end
 
            end
            
        end
        
        function localEvidence = makeLocalEvidence(model,obs)
        % localEvidence(i,t) = p(y(t) | S(t)=i)   
            
            localEvidence = zeros(model.nstates,size(obs,2));     
            for j = 1:model.nstates
                localEvidence(j,:) = exp(logprob(model.stateConditionalDensities{j},obs));
            end
            
        end
        
        
        function model = checkObservationModel(model,observationModel)
        % Basic check on the specified observation model - make sure its at least 
        % a PMTK class.
            switch(class(obserevationModel))
                case 'char'
                    obsmod = obseervationModel;
                otherwise
                    obsmod = class(observationModel);
            end
            
            try
                meta = metaclass(observationModel);
                
            catch
               error('%s is not a valid observation model. Make sure it names a valid PMTK distribution',obsmod);
            end
            model.observationModel = obsmod;
            
        end
        
        
        
        
            
            
            
            
            
        
        
%         function SS = expectedSSobservation(model,data,gamma)
%         % Compute the expected sufficient statistics for the observation model. 
%         
%         switch lower(model.observationModel)
%             case 'discrete'
%                 
%                 ss = zeros(model.nstates,model.noutputsymbols);
%                 for ex=1:numex
%                     obs = model.getObservation(data,ex);
%                     T = length(obs);  
%                     % loop over whichever is shorter
%                     if T < O
%                         for t=1:T
%                             o = obs(t);
%                             ss(:,o) = ss(:,o) + gamma(:,t);
%                         end
%                     else
%                         for o=1:O
%                             ndx = find(obs==o);
%                             if ~isempty(ndx)
%                                 ss(:,o) = ss(:,o) + sum(gamma(:, ndx), 2);
%                             end
%                         end
%                     end
%                 end
%             SS = struct('counts',{},'N',{});
%             for i=1:model.nstates
%                SS(i).counts = ss(i,:);
%                SS(i).N = N;
%             end
%             case 'mvn'
%                 
%             case 'mvnmix'
%                 
%         end
%         
%         end
        
        
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
                ndx = size(data,2)*ones(1,size(data,2)*size(data,3));
            end
        end
    end
end % end of class

