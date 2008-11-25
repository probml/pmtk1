classdef HmmDiscreteDist < HmmDist
   
    properties
        obsmat;          % The observation  matrix: 
                         % obsmat(i,j) = p(x_t = j | z_t = i)
                         % e.g. the probability of observing j at time t given
                         % that the hidden state is i at time t. 
                         
        transmat;        % The transisition matrix:
                         % transmat(i,j) = p(z_t = j | z_t-1 = i)
                         % e.g. the probability that the hidden state at time t
                         % is j given that the hidden state at time t-1 is i. 
                         
                       
        
        pi;              % Distribution over initial states
        nstates;         % The number of hidden states
        noutputSymbols;  % The number of output symbols 
    end
    
    methods
        
        function model = HmmDiscreteDist(varargin)
            [model.nstates,model.noutputs] = ...
                process_options(varargin,...
                'nstates'           ,[],...
                'noutputSymbols'    ,[]);
        end
        
        function model = fit(model,varargin)
            [X,pi0,obsmat0,transmat0,method,options] = process_options(varargin,...
                'X'        ,[]   ,...
                'pi0'      ,[]   ,...
                'obsmat0'  ,[]   ,...
                'transmat0',[]   ,...
                'method'   ,'em' );
             
            if(isempty(model.nstates) && ~isempty(transmat0))
                model.nstates = size(transmat0,1);
            end
            
            if(isempty(model.noutputSymbols) && ~isempty(obsmat0))
                model.noutputSymbols = size(obsmat0,2);
            end
            
            if(isempty(model.nstates) || isempty(model.noutputSymbols))
               error('Unknown number of hidden states and/or output symbols'); 
            end
            
            if(isempty(X))
               error('No data specified'); 
            end
            
            if(isempty(pi0))
               pi0 = normalise(rand(model.nstates,1));
            end
            
            if(isempty(obsmat0))
               obsmat0 = mk_stochastic(rand(model.nstates,model.noutputSymbols)); 
            end
            
            if(isempty(transmat0))
               transmat0 = mk+stochastic(rand(model.nstates)); 
            end
            
            switch(lower(method))
                case 'em'
                  [LL,obj.pi,obj.transmat,obj.obsmat]=  dhmm_em(X,pi0,transmat0,obsmat0,options{:});
                otherwise
                    error('%s is an unsupported fit method',method);
            end
            
        end
        
        function pred = predict(model,varargin)
           [X,target,indices] = process_options(varargin,'X',[],'target','z','indices',[]);
           
           
           
           switch(lower(target))
               case 'z'
                     %obslik(i,t) = Pr(Y(t)| Q(t)=i)
               %[alpha,beta,gamma] = fwdback(model.pi,model.transmat,model.obsmat,obslik);
               % pred = DiscreteProductDist(gamma');
               case 'x'
                   
               otherwise
                   error('%s is an unknown variable. Valid options include ''X'' or ''Z''',target);
           end
            
           
        end
        
        function lp = logprob(model,X)
            lp = zeros(size(X,1),1);
            if(iscell(X))
                for i=1:size(X,1)
                    obslik = multinomial_prob(X{i}, model.obsmat);
                    [alpha, beta, gamma, lp(i)] = fwdback(model.pi, model.transmat, obslik, 'fwd_only', 1);
                end
            else
                for i=1:size(X,1)
                    obslik = multinomial_prob(X(i,:), model.obsmat);
                    [alpha, beta, gamma, lp(i)] = fwdback(model.pi, model.transmat, obslik, 'fwd_only', 1);
                end
            end
        end
        
        function S = sample(model,nsamples,length)
            S = dhmm_sample(model.pi, model.transmat, model.obsmat, nsamples, length); 
        end
        
        function path = viterbi(model,X)
        % compute the viterbi path
            obslik = multinomial_prob(X,model.obsmat);
            path = viterbi_path(model.pi,model.transmat,obslik);
        end
        
        function d = ndims(model)
           d = model.nstates; 
        end
        
    end
    
end

