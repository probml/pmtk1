classdef MvnMixDist < MixtureDist
    
    
    
    methods
        
        function model = MvnMixDist(varargin)
           [nmixtures,mixingWeights,distributions,model.transformer] = process_options(varargin,'nmixtures',[],'mixingWeights',[],'distributions',[],'transformer',[]);
           if(isempty(mixingWeights) && ~isempty(nmixtures))
               mixingWeights = normalize(ones(1,nmixtures));
           end
           model.mixingWeights = mixingWeights;
           if(isempty(distributions))
               distributions = copy(MvnDist(),numel(model.mixingWeights),1);
           end
           model.distributions = distributions;
            
        end
        
       
        
        
        
    end
    
    methods(Access = 'protected')
        
        function displayProgress(model,data,loglik,rr)
            figure(1000);
            clf
            t = sprintf('RR: %d, negloglik: %g\n',rr,-loglik);
            fprintf(t);
            if(size(data,2) == 2)
                nmixtures = numel(model.distributions);
                plot(data(:,1),data(:,2),'.','MarkerSize',10);
                title(t);
                hold on;
                axis tight;
                for k=1:nmixtures
                    f = @(x)model.mixingWeights(k)*exp(logprob(model.distributions{k},x));
                    [x1,x2] = meshgrid(min(data(:,1)):0.1:max(data(:,1)),min(data(:,2)):0.1:max(data(:,2)));
                    z = f([x1(:),x2(:)]);
                    contour(x1,x2,reshape(z,size(x1)),'LevelList',0.6*max(z),'LineWidth',2);
                    mu = model.distributions{k}.mu;
                    plot(mu(1),mu(2),'rx','MarkerSize',15,'LineWidth',2);
                end
               
            end
        end
        
        
        
    end
    
    methods(Static = true)
        
        function testClass()
            
           load oldFaith;  m = fit(MvnMixDist('nmixtures',2,'transformer',StandardizeTransformer(true)),'data',X); 
            
           
           
           m = MvnMixDist();
           m = mkRndParams(m,2,2);
            
        end
        
    end
    
    
end