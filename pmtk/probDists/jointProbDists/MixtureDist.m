classdef MixtureDist < ParamJointDist
% A mixture model
% Mixtures of any probability distributions can be created with this class.
% Subclasses simply allow for more a more convienient interface by automatically
% instantiating the mixture components. To fit via EM, however, each mixture
% component must have a mkSuffStat method and a 'suffStat' option to fit(). 
    
    properties
        distributions;      % a cell array storing the distributions
        mixingWeights;      % DiscreteDist or Discrete_DirichletDist
        verbose = true;
        transformer;        % data preprocessor
        nrestarts = 5;      % number of random restarts
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
           if isempty(nmixtures)
               nmixtures = numel(distributions);
           end
           model.distributions = distributions;
           if(~isempty(nmixtures) && isempty(mixingWeights))
               mixingWeights = DiscreteDist('mu',normalize(ones(nmixtures,1)),'support',1:nmixtures);
           elseif(~isempty(model.distributions))
               mixingWeights = DiscreteDist('mu',normalize(ones(numel(model.distributions,1))),'support',1:nmixtures);
           end
           model.mixingWeights = mixingWeights;
        end
        
        function model = fit(model,varargin)
        % Fit via EM    
           [data,opttol,maxiter,nrestarts,SS,prior,init] = process_options(varargin,...
               'data'      ,[]    ,...
               'opttol'    ,1e-3  ,...
               'maxiter'   ,20    ,...
               'nrestarts' ,model.nrestarts ,...
               'suffStat', []     ,...
               'prior'     ,'none',...
               'init'      , true );
           nmixtures = numel(model.distributions);
           if(not(isempty(SS))),fitSS();return;end    
           if(~isempty(model.transformer))
              [data,model.transformer] = train(model.transformer,data); 
           end
           if(init),model = initializeEM(model,data);end
           bestDists = model.distributions;  
           bestMix   = model.mixingWeights; 
           bestLL    = sum(logprob(model,data)); 
           bestRR    = 1;
           for r = 1:nrestarts,
                emUpdate();
           end
           model.distributions = bestDists;
           model.mixingWeights = bestMix;
           if(model.verbose),displayProgress(model,data,bestLL,bestRR);end
           %% Sub functions to fit
            function fitSS()
            % Fit via sufficient statistics    
                for k=1:nmixtures
                    model.distributions{k} = fit(model.distributions{k},'suffStat',SS.ess{k},'prior',prior);
                end
                mixSS.counts = colvec(normalize(sum(SS.weights,1)));
                model.mixingWeights = fit(model.mixingWeights,'suffStat',mixSS);
            end
           
            function emUpdate()
            % Perform EM    
                if(r>1 && init),model = initializeEM(model,data);end
                converged = false; iter = 0; currentLL =sum(logprob(model,data));
                while(not(converged))
                    if(model.verbose),displayProgress(model,data,currentLL,r);end
                    prevLL = currentLL;
                    logRik = calcResponsibilities(model,data);  % responsibility of each cluster for each data point
                    Rik = exp(bsxfun(@minus,logRik,logsumexp(logRik,2))); % normalize
                    for k=1:nmixtures
                        ess = model.distributions{k}.mkSuffStat(data,Rik(:,k));
                        model.distributions{k} = fit(model.distributions{k},'suffStat',ess,'prior',prior); 
                    end
                    mixSS.counts = colvec(normalize(sum(Rik,1)));
                    model.mixingWeights = fit(model.mixingWeights,'suffStat',mixSS);
                    iter = iter + 1;
                    currentLL = sum(logprob(model,data));
                    converged = iter >=maxiter || (abs(currentLL - prevLL) / (abs(currentLL) + abs(prevLL) + eps)/2) < opttol;
                end
                if(currentLL > bestLL)
                    bestDists = model.distributions;
                    bestMix   = model.mixingWeights;
                    bestLL    = sum(logprob(model,data));
                    bestRR    = r;
                end   
                %%
            end % end of emUpdate() subfunction
        end % end of fit method
        
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
        
        
        function Tfac = convertToTabularFactor(model, globalDomain,visVars,visVals)
        % globalDomain = indices of each parent, followed by index of child
        % all of the children must be observed
            if(isempty(visVars)) 
               Tfac = TabularFactor(1,globalDomain); return; % return an empty TabularFactor
            end
            if ~isequal(globalDomain(2:end),visVars)
                error('Not all of the continuous valued children of this CPD were observed.');
            end
            T = exp(calcResponsibilities(model,visVals));
            Tfac = TabularFactor(T,globalDomain(1)); % only a factor of the parent now
        end
        
        
        function model = mkRndParams(model, d,K)
            for i=1:K
                model.distributions{i} = mkRndParams(model.distributions{i},d);
            end
            model.mixingWeights = DiscreteDist('mu',normalize(rand(K,1)));
        end
        
        function model = condition(model, visVars, visValues)
        % pass condition requests through to mixture components
            if nargin < 2
                visVars = []; visValues = [];
            end
            model.conditioned = true;
            model.visVars = visVars;
            model.visVals = visValues;
            for i=1:numel(model.distributions)
                model.distributions{i} = condition(model.distributions{i},visVars,visValues);
            end
        end
         
        function postQuery = marginal(model, queryVars)
            % keep only the queryVars mixture components - barren node removal
            if(numel(queryVars == 1))
                postQuery = model.distributions{queryVars};
            else
                model.distributions = model.distributions{queryVars};
                model.mixingWeights = marginal(model.mixingWeights,queryVars);
                postQuery = model;
            end
        end
         
         function S = sample(model,nsamples)
              if nargin < 2, nsamples = 1; end
              Z = sampleDiscrete(mean(model.mixingWeights)', nsamples, 1);
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
         
         function d = ndistrib(model)
            d = max(1,numel(model.distributions)); 
         end
         
         
         function SS = mkSuffStat(model,data,weights) 
         % Compute weighted, (expected) sufficient statistics. In the case of 
         % an HMM, the weights correspond to gamma = normalize(alpha.*beta,1)
         % We calculate gamma2 by combining alpha and beta messages with the
         % responsibilities - see equation 13.109 in pml24nov08
             if(nargin < 2)
                 weights = ones(size(data,1));
             end
             if(~isempty(model.transformer))
                 [data,model.transformer] = train(model.transformer,data);
             end
             logRik = calcResponsibilities(model,data);
             logGamma2 = bsxfun(@plus,logRik,log(weights+eps));           % combine alpha,beta,local evidence messages
             %logGamma2 = bsxfun(@minus,logGamma2,logsumexp(logGamma2,2)); % normalize while avoiding numerical underflow
             logGamma2 = normalizeLogspace(logGamma2);
             gamma2 = exp(logGamma2);
             nmixtures = numel(model.distributions);
             ess = cell(nmixtures,1);
             for k=1:nmixtures
                 ess{k} = model.distributions{k}.mkSuffStat(data,gamma2(:,k));   
             end
             SS.ess = ess;
             SS.weights = gamma2;
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
            if(model.verbose)
                fprintf('RR: %d, negloglik: %g\n',rr,-loglik);
            end
        end
        
        function logRik = calcResponsibilities(model,data)
        % returns unnormalized log responsibilities
        % logRik(i,k) = log(p(data(n,:),Z_k | params))
        % Used by predict(), logprob(), mkSuffStat()
            if(~isempty(model.transformer))
                data = test(model.transformer,data);
            end
            n = size(data,1); nmixtures = numel(model.distributions);
            logRik = zeros(n,nmixtures);
            for k=1:nmixtures
                logRik(:,k) = log(sub(mean(model.mixingWeights),k)+eps)+sum(logprob(model.distributions{k},data),2); 
                % Calling logprob on vectorized distributions, (representing a
                % product, e.g. product of Bernoulli's) returns a matrix. We
                % therefore sum along the 2nd dimension in the 2nd term. This
                % has no effect for other distributions as logprob returns a
                % column vector. 
            end
        end
        
        
    end
    
  
        
    
    
end

