classdef MvnMixDist < MixtureDist
    
    
    
    methods
        
        function model = MvnMixDist(varargin)
           [nmixtures,mixingWeights,distributions,model.transformer] = process_options(varargin,...
               'nmixtures',[],'mixingWeights',[],'distributions',[],'transformer',[]);
           if(isempty(mixingWeights) && ~isempty(nmixtures))
               mixingWeights = normalize(ones(1,nmixtures));
           end
           model.mixingWeights = mixingWeights;
           if(isempty(distributions))
               distributions = copy(MvnDist(),numel(model.mixingWeights),1);
           end
           model.distributions = distributions;
            
        end
        
        
         function model = mkRndParams(model, d,K)
            model.distributions = copy(MvnDist(),K,1);
            model = mkRndParams@MixtureDist(model,d,K);
             
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
                if(nmixtures == 2)
                    colors = subd(predict(model,data),'mu')';
                    scatter(data(:,1),data(:,2),18,[colors(:,1),zeros(size(colors,1),1),colors(:,2)],'filled');
                else
                    plot(data(:,1),data(:,2),'.','MarkerSize',10);
                end
                
                
                title(t);
                hold on;
                axis tight;
                for k=1:nmixtures
                    f = @(x)model.mixingWeights(k)*exp(logprob(model.distributions{k},x));
                    [x1,x2] = meshgrid(min(data(:,1)):0.1:max(data(:,1)),min(data(:,2)):0.1:max(data(:,2)));
                    z = f([x1(:),x2(:)]);
                    contour(x1,x2,reshape(z,size(x1)));
                    mu = model.distributions{k}.mu;
                    plot(mu(1),mu(2),'rx','MarkerSize',15,'LineWidth',2);
                end
               
            end
        end
        
        
        
    end
    
    methods(Static = true)
        
        function testClass()
           if(1) 
            %dists = {MvnDist([1,-1],0.1*eye(2)),MvnDist([-1,1],0.1*eye(2
            setSeed(0);
            load oldFaith;  
            %m = fit(MvnMixDist('nmixtures',2,'distributions',dists,'transformer',StandardizeTransformer(false)),'data',X,'init',false,'nrestarts',1); 
            m = fit(MvnMixDist('nmixtures',2,'transformer',StandardizeTransformer(false)),'data',X); 
            pred = predict(m,X);
           end
           if(0)
           setSeed(13);
           m = mkRndParams(MvnMixDist(),2,4);
           plot(m);
           X = sample(m,1000);
           hold on;
           plot(X(:,1),X(:,2),'.','MarkerSize',10);
           m1 = fit(MvnMixDist('nmixtures',4),'data',X);
           end
         
            
        end
        
    end
    
    
end