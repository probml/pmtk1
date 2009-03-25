classdef MarkovDist < ParamDist
    
    properties
        startDist;                  % distribution over starting states: 
                                    % a DiscreteDist 

        transitionDist;             % p( S(t) = j | S(t-1) = i),
                                    % A DiscreteDist (vectorized) 
                                    % (Each *column* of 
                                    % mean(transitionDist) sums to one)
    
        support;                    % the discrete values each state can take on
    
    end
    
    
    methods
        
        function model = MarkovDist(startDist,transitionDist,support)
            if nargin < 1, startDist      = DiscreteDist(); end
            if nargin < 2, transitionDist = DiscreteDist(); end
            model.startDist = startDist;
            model.transitionDist = transitionDist;
            if nargin < 3 || isempty(support)
               support = model.transitionDist.support; 
            end
            model.support = support; 
        end
        
        
        function model = fit(model,X)
            % X may be an n-by-d matrix or an n-by-1 cell array
            
            if isempty(model.support)
               if iscell(X), model.support = unique(cell2mat(X'));
               else          model.support = unique(X(:)); end
            end
            map = @(x)canonizeLabels(x,model.support);
            if iscell(X), model.startDist = fit(model.StartDist,'data',cellfun(@(c)c(1),X));
            else          model.startDist = fit(model.startDist,'data',X(:,1)); end
            T = zeros(numel(model.support));
            if iscell(X)
                for i=1:numel(X)
                    Xi = map(X{i});
                    for j=1:length(Xi)-1
                        T(Xi(j),Xi(j+1)) = T(Xi(j),Xi(j+1)) + 1;
                    end
                end
            else
                X = canonizeLabels(X,model.support);
                for i=1:size(X,1);
                    Xi = X(i,:);
                    for j=1:length(Xi)-1
                        T(Xi(j),Xi(j+1)) = T(Xi(j),Xi(j+1)) + 1;
                    end
                end
            end
            SS.counts = T';
            model.transitionDist = fit(model.transitionDist,'suffStat',SS);
        end
        
        function logp = logprob(model,X)
            map = @(x)canonizeLabels(x,model.support);
            if iscell(X), X = cellfun(@(c)map(c),X);
            else          X = map(X); end
            N = size(X,1);
            logp = zeros(N,1);
            logPi = log(pmf(model.startDist)+eps);
            logT  = log(pmf(model.transitionDist)'+eps);
            for i=1:N
               if iscell(X),  Xi = X{i};
               else           Xi = X(i,:);   end
               nstates = numel(model.support);
               Njk = zeros(nstates);
               for j=1:nstates-1
                  Njk(Xi(j),Xi(j+1)) =  Njk(Xi(j),Xi(j+1)) + 1;
               end
               logp(i) = logPi(Xi(1)) + sumv(Njk.*logT,[1,2]);
            end
            
            
        end
            
        function X = sample(model,len,n)
            if nargin < 3, n = 1; end
             X = model.support(mc_sample(pmf(model.startDist)',pmf(model.transitionDist)',len,n));
        end
        
        function pi = stationaryDistribution(model)
           K = numel(model.support);
           T = pmf(model.transitionDist)';
           pi = DiscreteDist((ones(1,K) / (eye(K)-T+ones(K)))');
        end
    
        
    end
    
    
    
    
    methods(Static = true)
        
        function testClass()
            setSeed(0);
            sourceDist = MarkovDist(DiscreteDist(normalize(rand(3,1))),DiscreteDist(normalize(rand(3),1)),[5,6,7]);
            X = sample(sourceDist,100,100);
            testDist = fit(MarkovDist,X);
            
            pmf(sourceDist.startDist)
            pmf(testDist.startDist)
            pmf(sourceDist.transitionDist)
            pmf(testDist.transitionDist)
            
            
        end
        
    end
    
    
    
    
end