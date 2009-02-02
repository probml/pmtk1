classdef MvnMixDist < MixtureDist
% Mixture of Multivariate Normal Distributions    
    
    
    methods
        
        function model = MvnMixDist(varargin)
           [nmixtures,mixingWeights,distributions,model.transformer,model.verbose,model.nrestarts] = process_options(varargin,...
               'nmixtures',[],'mixingWeights',[],'distributions',[],'transformer',[],'verbose',true,'nrestarts',model.nrestarts);
           if(~isempty(distributions)),nmixtures = numel(distributions);end
               
           if(isempty(mixingWeights) && ~isempty(nmixtures))
               mixingWeights = DiscreteDist('mu',normalize(ones(nmixtures,1)));
           end
           model.mixingWeights = mixingWeights;
           if(isempty(distributions)&&~isempty(model.mixingWeights))
               distributions = copy(MvnDist(),nstates(model.mixingWeights),1);
           end
           model.distributions = distributions;
            
        end
        
         function model = mkRndParams(model, d,K)
            model.distributions = copy(MvnDist(),K,1);
            model = mkRndParams@MixtureDist(model,d,K);
         end
         
         function mu = mean(m)
             mu = zeros(ndimensions(m),ndistrib(m));
             for k=1:ndistrib(m)
                 mu(:,k) = colvec(m.distributions{k}.mu);
             end
             
             M = bsxfun(@times,  mu, rowvec(mean(m.mixingWeights)));
             mu = sum(M, 2);
         end
    
         function C = cov(m)
             mu = mean(m);
             C = mu*mu';
             for k=1:numel(m.distributions)
                 mu = m.distributions{k}.mu;
                 C = C + sub(mean(m.mixingWeights),k)*(m.distributions{k}.Sigma + mu*mu');
             end
         end
         
         
         function xrange = plotRange(obj, sf)
             if nargin < 2, sf = 3; end
             %if ndimensions(obj) ~= 2, error('can only plot in 2d'); end
             mu = mean(obj); C = cov(obj);
             s1 = sqrt(C(1,1));
             x1min = mu(1)-sf*s1;   x1max = mu(1)+sf*s1;
             if ndimensions(obj)==2
                 s2 = sqrt(C(2,2));
                 x2min = mu(2)-sf*s2; x2max = mu(2)+sf*s2;
                 xrange = [x1min x1max x2min x2max];
             else
                 xrange = [x1min x1max];
             end
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
                    f = @(x)sub(mean(model.mixingWeights),k)*exp(logprob(model.distributions{k},x));
                    [x1,x2] = meshgrid(min(data(:,1)):0.1:max(data(:,1)),min(data(:,2)):0.1:max(data(:,2)));
                    z = f([x1(:),x2(:)]);
                    contour(x1,x2,reshape(z,size(x1)));
                    mu = model.distributions{k}.mu;
                    plot(mu(1),mu(2),'rx','MarkerSize',15,'LineWidth',2);
                end
               
            end
        end
        
        
        
    end
    
  
    
end