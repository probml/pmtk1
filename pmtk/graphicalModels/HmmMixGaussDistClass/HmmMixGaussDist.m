classdef HmmMixGaussDist < HmmDist
    
    properties
        transmat;
        mixmat;
        pi;
        mu;                     
        Sigma;
        nstates;
        nmixtures;
    end
    
    methods
        
        function model = HmmMixGaussDist(varargin)
            [model.nstates,model.nmixtures] = process_options(varargin,'nstates',[],'nmixtures',[]);
        end
        
        function model = fit(model,varargin)
            [X,pi0,transmat0,mu0,Sigma0,mixmat0,method,options] = process_options(varargin,...
                'X'         ,[], ...
                'pi0'       ,[],...
                'transmat0' ,[],...
                'mu0'       ,[],...
                'Sigma0'    ,[],...
                'mixmat0'   ,[],...
                'method'    ,'em');
            
   
                
            
            switch(lower(method))
                case 'em'
                    [LL, model.pi, model.transmat, model.mu, model.Sigma, model.mixmat] = mhmm_em...
                        (X, pi0, transmat0, mu0, Sigma0, mixmat0,options{:});
                otherwise
                    error('%s is not a supported fit method',method);
                    
            end
        end
        
        function pred = predict(model,varargin)
           [X,target,indices,method] = process_options(varargin,'X',[],'target','z','indices',[],'method','smoothing');
           if(~isempty(indices)),error('not yet implemented');end
           switch(lower(target))
               case 'z'
               obslik = mixgauss_prob(X, model.mu, model.Sigma, model.mixmat);      
               [alpha,beta,gamma] = fwdback(model.pi,model.transmat,model.obsmat,obslik);
               switch lower(method)
                   case 'smoothing'
                       pred = DiscreteProductDist(gamma');
                   case 'filtering'
                       pred = DiscreteProductDist(alpha');     
                   otherwise 
                       error('%s is an unsupported method',method);
               end
               case 'x'
                  error('not yet implemented'); 
               otherwise
                   error('%s is an unknown variable. Valid options include ''X'' or ''Z''',target);
                   
           end
            
            
            
        end
        
        function path = viterbi(model,X)
            obslik = mixgauss_prob(X, model.mu, model.Sigma, model.mixmat);
            path = viterbi_path(model.pi, model.transmat, obslik);
        end
        
        function logp = logprob(model,X)
            
            if(iscell(X))
                nexamples = numel(X);
                logp = zeros(nexamples,1);
                for i=1:nexamples
                    logp(i) = mhmm_logprob(X{i}, model.pi, model.transmat, model.mu, model.Sigma, model.mixmat);
                end
            else
                logp = zeros(size(X,3));
                for i=1:size(X,3)
                    logp(i) = mhmm_logprob(X(:,:,i),model.pi, model.transmat, model.mu, model.Sigma, model.mixmat);
                end
            end
            
        end
        
        function [obs,hidden] = sample(model,nsamples,length)
           [obs, hidden] = mhmm_sample(length, nsamples, initial_prob, model.transmat, model.mu, model.Sigma, model.mixmat);
        end
        
        function d = ndims(model)
            d = size(model.mu,1);
        end
        
     
        
    end
    
    
    methods(Static = true)
        
        
        function testClass()
            load data45;
            nstates   = 5;
            obsdims   = 13;
            nmix      = 1;
            pi0 = [1,0,0,0,0]';
            mu0 = randn(obsdims,nstates,nmix);
            Sigma0 = repmat(eye(obsdims),[1 1 nstates]);
            transmat0 = mk_stochastic(diag(ones(nstates,1)) + diag(ones(nstates-1,1),1)); 
            model4 = HmmMixGaussDist('nstates',5,'nmixtures',nmix);
            model4 = fit(model4,'X',train4,'pi0',pi0,'mu0',mu0,'Sigma0',Sigma0,'transmat0',transmat0,'max_iter',20);
            
            model5 = HmmMixGaussDist('nstates',5,'nmixtures',nmix);
            model5 = fit(model4,'X',train5,'pi0',pi0,'mu0',mu0,'Sigma0',Sigma0,'transmat0',transmat0,'max_iter',20);
            
            logp4 = logprob(model4,test45);
            logp5 = logprob(model5,test45);
            [val,yhat] = max([logp4,logp5],[],2);
            yhat(yhat == 1) = 4;
            yhat(yhat == 2) = 5;
            nerrs = sum(yhat ~= labels');
            
            if(exist('specgram','file'))
                subplot(2,2,1);
                specgram(signal1); 
               
                subplot(2,2,2)
                specgram(signal2);
              
                subplot(2,2,3);
                plot(viterbi(model4,mfcc1));
                subplot(2,2,4);
                plot(viterbi(model4,mfcc2));
                maximizeFigure;
                
            end
            
            
            
            if(0)
                O = 13;
                T = 50;
                nex = 50;
                data = randn(O,T,nex);
                M = 1;
                Q = 5;
                left_right = 0;
                prior0 = normalise(rand(Q,1));
                transmat0 = mk_stochastic(rand(Q,Q));
                [mu0, Sigma0] = mixgauss_init(Q*M, reshape(data, [O T*nex]), 'full');
                mu0 = reshape(mu0, [O Q M]);
                Sigma0 = reshape(Sigma0, [O O Q M]);
                mixmat0 = mk_stochastic(rand(Q,M));
                [LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
                    mhmm_em(data, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 2);
            end
        end
        
        
        
        
    end
    
end

