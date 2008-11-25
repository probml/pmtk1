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
            [model.nstates,model.obsmat,model.transmat,model.pi,model.noutputSymbols] = ...
                process_options(varargin,...
                'nstates'           ,[],...
                'obsmat'            ,[],...
                'transmat'          ,[],...
                'pi'                ,[],...
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
           [X,target,indices,method] = process_options(varargin,'X',[],'target','z','indices',[],'method','smoothing');
           
           if(~isempty(indices)),error('not yet implemented');end
           if(iscell(X)),error('cell arrays not yet implemented');end
           switch(lower(target))
               case 'z'
                   switch lower(method)
                       
                       case 'smoothing'
                           pred = zeros(size(X,1),model.nstates,size(X,2));
                           for i=1:size(X,1)
                               obslik = multinomial_prob(X(i,:),model.obsmat);
                               [alpha,beta,gamma] = fwdback(model.pi,model.transmat,obslik);
                               pred(i,:,:) = gamma;
                           end
                       case 'filtering'
                           pred = zeros(size(X,1),model.nstates,size(X,2));
                           for i=1:size(X,1)
                               obslik = multinomial_prob(X(i,:),model.obsmat);
                               [alpha,beta,gamma] = fwdback(model.pi,model.transmat,obslik);
                               pred(i,:,:) = alpha;
                           end
                      
                       otherwise
                           error('%s is not a valid method',method);
                   end
                       
                   
                     %obslik(i,t) = Pr(Y(t)| Q(t)=i)
               %[alpha,beta,gamma] = fwdback(model.pi,model.transmat,model.obsmat,obslik);
               % pred = DiscreteProductDist(gamma');
               case 'x'
                   error('not yet implemented');
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
        
        function [obs,hidden] = sample(model,nsamples,length)
            [obs,hidden] = dhmm_sample(model.pi, model.transmat, model.obsmat, nsamples, length); 
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
    
    methods(Static = true)
        
        function casino()
            % 1 = fair, 2 = loaded
            obsmat = [ones(1,6)./6;ones(1,5)./10,0.5];
            %transmat = [0.95,0.05;0.1,0.90];
            transmat  = [0.99,0.01;0.1,0.90];
            pi = [0.5,0.5];
            model = HmmDiscreteDist('nstates',2,'noutputSymbols',6,'obsmat',obsmat,'transmat',transmat,'pi',pi);
            nsamples = 300; length = 1;
            [rolls,die] = sample(model,nsamples,length);
            dielabel = repmat('F',size(die));
            dielabel(die == 2) = 'L';
            vit = zeros(size(die));
            for i=1:nsamples
                vit(i,:) = viterbi(model,rolls(i,:));
            end
            vitlabel = repmat('F',size(vit));
            vitlabel(vit == 2) = 'L';
            rollLabel = num2str(rolls);
            
            for i=1:60:300
                fprintf('Rolls:\t %s\n',rollLabel(i:i+59));
                fprintf('Die:\t %s\n',dielabel(i:i+59));
                fprintf('Viterbi: %s\n\n',vitlabel(i:i+59));
            end
            
            filtered = predict(model,'X',rolls,'method','filtering');
            smoothed = predict(model,'X',rolls,'method','smoothing');
            d = 1:300;
            figure; hold on;
            plot(d(die == 1),0,'.r','MarkerSize',10);
            
            
            plot(filtered(:,1));
            title('filtered');
            %line(repmat(fair,1,2)',[zeros(nfair,1),ones(nfair,1)]','color','k');
            
            figure;
            plot(smoothed(:,2));
            title('smoothed');
            
            
            
        end
        
        
        
    end
    
end

