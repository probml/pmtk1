classdef HmmDist < ProbDist
% Hidden Markov Model   
    
    properties                           
        nstates;                    % number of hidden states

        pi;                         % initial/starting distribution over hidden states 

        transitionMatrix;           % matrix of size nstates-by-nstates
                                    % transitionMatrix(i,j) = p( Z(t) = j | Z(t-1) = i), 
                                    % where Z(t) is the the hidden state at time t.
                                                 
        stateConditionalDensities;  % the observation model - one object per 
                                    % hidden state stored as a cell array.
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
        % 'observationModel' - a string describing the observation model to use: 
        %                      'discrete' | 'mvn' | 'mixMvn'
        %
        % 'method'            - ['map'] | 'bayesian'
        %
        % 'algorithm'        -  ['em']  the fitting algorithm to use
        %
        % 'transitionPrior'  - either a single DirichletDist object used as a
        %                      prior for each row of the transition matrix or a
        %                      cell array of DirichletDist objects, one for each
        %                      row. 
        %
        % 'observationPrior' - a single object acting as the prior for each
        %                      stateConditionalDensity.
        %
        % Any additional arguments are passed directly to the implementation of the
        % fit algorithm.
        %
        % If model.stateConditionalDensities is non-empty, these objects are
        % used to initialize the fitting algorithm. Similarly for model.pi and
        % model.transitionMatrix.
            
             [data,method,algorithm,observationModel,transitionPrior,observationPrior,options] = process_options(varargin,...
                 'data'             ,[]         ,...
                 'method'           ,'map'      ,...
                 'algorithm'        ,'em'       ,...
                 'observationModel' ,''         ,...
                 'transitionPrior'  ,[]         ,...
                 'observationPrior' ,[]         );
             
                 
             
             data = checkData(model,data);
             
             
             switch lower(observationModel)
                 case 'discrete'
                     switch lower(method)
                         case 'map'
                            switch lower(algorithm)
                                case 'em'
                                    emUpdateDiscrete(model,data,transitionPrior,observationPrior,options{:});
                                otherwise
                                    error('%s is not a valid map algorithm',algorithm);
                            end
                         case 'bayesian'
                                    
                             
                         otherwise
                             error('%s is not a valid fit method',method);
                     end
                     
                 case 'mvn'
                     error('not yet implemented');
                 case 'mvnMix'
                     error('not yet implemented');
                 otherwise
                     if(isempty(observationModel))
                       error('You must specify an observation model: currently one of ''discrete'' | ''mvn'' | ''mvnMix''');  
                     else
                        error('%s is not a supported observation model',observationModel); 
                     end
             end
             
             
                
        end

        
        
        
        
        function logp = logprob(model,X)
            
            
        end
        
        
        function pred = predict(model,varargin)
            
        end
        
        function [obs,hidden] = sample(model,nsamples,length)
            
        end
        
        function d = ndims(model)
            d = model.obsDims;
        end
        
        function path = viterbi(model,X)
            
        end
        
    end
    
    methods(Access = 'protected')
        
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
        
        function emUpdate(model,data,transitionPrior,observationPrior,varargin)
        % Update the transition matrix, the state conditional densities and pi,
        % the distribution over starting hidden states using em.
        
               [opt_tol,max_iter,clamp_pi,clamp_obs,clamp_trans] = ...
                   process_options(varargin ,...
                   'opt_tol'                ,1e-4   ,...
                   'max_iter'               ,20     ,...
                   'clamp_pi'               ,false  ,...
                   'clamp_obs'              ,false  ,...
                   'clamp_trans'            ,false  );
           
               loglikelihood = 0;
               iter = 0;
               converged = false;
               switch(class(data))
                   case 'cell'
                       nobservations = numel(data);
                   case 'double'
                       nobservations = size(data,3);
               end
               
               
               while(iter < max_iter && ~converged)
                   
                   
                   exp_num_visits1 = zeros(model.nstates,1);
                   exp_num_trans   = zeros(model.nstates,model.nstates);
                   for j=1:nobservations
                         switch(class(data))
                            case 'cell'
                            	observation = data{j};
                            case 'double'
                                observation = data(:,:,j);
                        end
                        obslength = size(observation,2);
                        obslik = zeros(model.nstates,obslength);
                        for i=1:obj.nstates
                           stateCondDist = model.stateConditionalDensities{i};
                           obslik(:,i) = exp(logprob(stateCondDist,observation)); 
                        end
                        [gamma, alpha, beta, current_ll] = hmmFwdBack(model.pi, model.transitionMatrix, obslik);
                        exp_num_visits1 = exp_num_visits1 + gamma(:,1);
                        xi = hmmComputeTwoSlice(alpha, beta, model.transitionMatrix, obslik);
                        exp_num_trans   = exp_num_trans + xi;
                        
                                
                        loglikelihood = loglikelihood + current_ll;
                   end
                   
                  
                   
                   
                   
               end % end of em loop
            
        end
        
    end % end of protected methods
end % end of class

