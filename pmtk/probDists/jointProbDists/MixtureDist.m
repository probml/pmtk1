classdef MixtureDist < ParamJointDist
% A mixture model   
    
    properties
        distributions;      % a cell array storing the distributions
        mixingWeights;      % pi
        verbose = true;
        transformer;        % data preprocessor
    end
    
    methods
        
        function model = MixtureDist(varargin)
        % Construct a new mixture of distributions    
           [nmixtures,distributions,mixingWeights,model.transformer]...
               = process_options(varargin,...
               'nmixtures'    ,[] ,...
               'distributions',[] ,...
               'mixingWeights',[] ,...
               'transformer'  ,[]);
           
           if ~isempty(nmixtures) && numel(distributions) == 1
               distributions = copy(distributions,nmixtures,1);
           end
           model.distributions = distributions;
           if(~isempty(nmixtures) && isempty(mixingWeights))
               mixingWeights = normalize(ones(1,nmixtures));
           elseif(~isempty(model.distributions))
               mixingWeights = normalize(ones(1,numel(model.distributions)));
           end
           model.mixingWeights = mixingWeights;
        end
        
        function model = fit(model,varargin)
        % Fit via EM    
           [data,opttol,maxiter,nrestarts,prior,init] = process_options(varargin,...
               'data'      ,[]    ,...
               'opttol'    ,1e-3  ,...
               'maxiter'   ,20    ,...
               'nrestarts' ,5     ,...
               'prior'     ,'none',...
               'init'      , true );
           if(~isempty(model.transformer))
              [data,model.transformer] = train(model.transformer,data); 
           end
           if(init),model = initializeEM(model,data);end
           n = size(data,1); nmixtures = numel(model.distributions);
           bestDists = model.distributions;
           bestMix = model.mixingWeights;
           bestLL = sum(logprob(model,data)); bestRR = 1;
           for r = 1:nrestarts
               if(r>1 && init),model = initializeEM(model,data);end
               converged = false; iter = 0; currentLL =sum(logprob(model,data));
               while(not(converged))
                   if(model.verbose),displayProgress(model,data,currentLL,r);end
                   prevLL = currentLL;
                   pred = predict(model,data);
                   Rik = pred.mu';                   
                   for k=1:nmixtures
                       ess = model.distributions{k}.mkSuffStat(data,Rik(:,k));
                       model.distributions{k} = fit(model.distributions{k},'suffStat',ess,'prior',prior);
                   end
                   model.mixingWeights = sum(Rik,1)./n;
                   iter = iter + 1;
                   currentLL = sum(logprob(model,data));
                   converged = iter >=maxiter || (abs(currentLL - prevLL) / (abs(currentLL) + abs(prevLL) + eps)/2) < opttol;
               end
               if(currentLL > bestLL)
                   bestDists = model.distributions;
                   bestMix = model.mixingWeights;
                   bestLL =  sum(logprob(model,data));
                   bestRR = r;
               end
           end
           model.distributions = bestDists;
           model.mixingWeights = bestMix;
           if(model.verbose),displayProgress(model,data,bestLL,bestRR);end
        end
        
        function pred = predict(model,data)
        % pred.mu(k,i) = p(Z_k | data(i,:),params)   
            logRik = calcResponsibilities(model,data);
            Rik = exp(bsxfun(@minus,logRik,logsumexp(logRik,2))); 
            pred = DiscreteDist('mu',Rik');
        end
         
        function logp = logprob(model,data)
        % logp(i) = log p(data(i,:) | params) 
              logp = logsumexp(calcResponsibilities(model,data),2);
        end
        
        function model = mkRndParams(model, d,K)
            for i=1:K
                model.distributions{i} = mkRndParams(model.distributions{i},d);
            end
            model.mixingWeights = normalize(rand(1,K));
        end
        
         function model = condition(model, visVars, visValues)
            for i=1:numel(model.distributions)
               model.distributions{i} = condition(model.distributions{i},visVars,visValues); 
            end
         end
         
         function postQuery = marginal(model, queryVars)
             model.distributions = model.distributions{queryVars};
             model.mixingWeights = model.mixingWeights(queryVars);
             postQuery = model;
         end
         
         function S = sample(model,nsamples)
              if nargin < 2, nsamples = 1; end
              Z = sampleDiscrete(model.mixingWeights, nsamples, 1);
              d = ndimensions(model);
              S = zeros(nsamples, d);
              for i=1:nsamples
                 S(i,:) = rowvec(sample(model.distributions{Z(i)}));
              end
         end
         
         function d = ndimensions(model)
            if(numel(model.distributions) > 0)
                d = ndimensions(model.distributions{1}); 
            else
                d = 0;
            end
         end
         
         function h = plot(model,varargin)
            h = figure; hold on;
            for i=1:numel(model.distributions)
               plot(model.distributions{i},'scaleFactor',model.mixingWeights(i),varargin{:}); 
            end
         end
       
    end
    
    
    methods(Access = 'protected')
        
        function model = initializeEM(model,X)
        % override in subclass if necessary    
          for k=1:numel(model.distributions)
             model.distributions{k} = mkRndParams(model.distributions{k},X); 
          end
        end
        
        function displayProgress(model,data,loglik,rr)
        % override in subclass to customize displayed info    
            fprintf('RR: %d, negloglik: %g\n',rr,-loglik);
        end
        
        function logRik = calcResponsibilities(model,data)
        % returns unnormalized log responsibilities
        % logRik(i,k) = log(p(data(n,:),Z_k | params))
        % Used by predict() and logprob()
            if(~isempty(model.transformer))
                data = test(model.transformer,data);
            end
            n = size(data,1); nmixtures = numel(model.distributions);
            logRik = zeros(n,nmixtures);
            for k=1:nmixtures
                logRik(:,k) = log(model.mixingWeights(k))+sum(logprob(model.distributions{k},data),2);
            end
        end
        
        
    end
    
    
  
        
        
    
    
end

