classdef MixtureDist < ParamJointDist
   
    
    properties
        distributions;
        mixingWeights;
        verbose = true;
        transformer;
    end
    
    methods
        
        function model = MixtureDist(varargin)
           [nmixtures,distributions,mixingWeights,model.transformer] = process_options(varargin,'nmixtures',[],'distributions',[],'mixingWeights',[],'transformer',[]);
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
           [data,opttol,maxiter,prior] = process_options(varargin,'data',[],'opttol',1e-10,'maxiter',100,'prior','none');
           if(~isempty(model.transformer))
              [data,model.transformer] = train(model.transformer,data); 
           end
           n = size(data,1);
           nmixtures = numel(model.distributions);
           model = initializeEM(model,data);
           converged = false;
           iter = 0;
           currentLL = 0;
           while(not(converged))
                if(model.verbose)
                    if(size(data,2) == 2)
                       plot(data(:,1),data(:,2),'.');
                       t = sprintf('negloglik = %g\n',-currentLL);
                       fprintf(t);
                       title(t);
                       hold on;
                       for k=1:nmixtures
                          plot(model.distributions{k});
                       end
                       pause(0.5);
                    end
               end
               prevLL = currentLL;
               Rik = zeros(n,nmixtures);  % responsibilities
               for k=1:nmixtures
                   Rik(:,k) = model.mixingWeights(k)*exp(logprob(model.distributions{k},data));
               end
               Rik = normalize(Rik,2);
               for k=1:nmixtures
                   model.distributions{k} = fit(model.distributions{k},'suffStat',model.distributions{k}.mkSuffStat(data,Rik(:,k)),'prior',prior);
               end
               
               model.mixingWeights = sum(Rik,1)./n;
               iter = iter + 1;
               currentLL = sum(logprob(model,data));
              
               converged = iter >=maxiter || (abs(currentLL - prevLL) / (abs(currentLL) + abs(prevLL) + eps)/2) < opttol;
               if(model.verbose && ~converged)
                  clf; 
               end
           end
        end
        
        function logp = logprob(model,data)
            p = zeros(size(data,1),1);
            for k = 1:numel(model.distributions)
                p = p + exp(logprob(model.distributions{k},data))*model.mixingWeights(k);
            end
            logp = log(p);
        end
       
    end
    
    
    methods(Access = 'protected')
        
        function model = initializeEM(model,X)
           for k=1:numel(model.distributions)
              model.distributions{k} = mkRndParams(model.distributions{k},size(X,2));
           end
          
        end
        
        
    end
    
    
    methods(Static = true)
        
        function testClass()
           load oldFaith;
           model = MixtureDist('nmixtures',2,'distributions',MvnDist(),'transformer',StandardizeTransformer(true));
           model = fit(model,'data',X);
        end
        
        
        
        
    end
    
end

