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
           [data,opttol,maxiter,nrestarts,prior] = process_options(varargin,...
               'data'      ,[]     ,...
               'opttol'    ,1e-5   ,...
               'maxiter'   ,100    ,...
               'nrestarts' ,20     ,...
               'prior'     ,'none' );
           if(~isempty(model.transformer))
              [data,model.transformer] = train(model.transformer,data); 
           end
           model = initializeEM(model,data);
           n = size(data,1); nmixtures = numel(model.distributions);
           bestDists = model.distributions;
           bestMix = model.mixingWeights;
           bestLL = sum(logprob(model,data)); bestRR = 1;
           for r = 1:nrestarts
               if(r>1),model = initializeEM(model,data);end
               converged = false; iter = 0; currentLL = -inf;
               while(not(converged))
                   if(model.verbose),displayProgress(model,data,currentLL,r);end
                   prevLL = currentLL;
                   Rik = zeros(n,nmixtures);  % responsibilities
                   for k=1:nmixtures
                       Rik(:,k) = model.mixingWeights(k)*exp(logprob(model.distributions{k},data));
                   end
                   Rik = normalize(Rik,2);
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
        
        function logp = logprob(model,data)
        % logp(i) = log p(data(i,:) | params)    
            p = zeros(size(data,1),1);
            for k = 1:numel(model.distributions)
                p = p + exp(logprob(model.distributions{k},data))*model.mixingWeights(k);
            end
            logp = log(p);
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
        
        
    end
    
    
  
        
        
    
    
end

