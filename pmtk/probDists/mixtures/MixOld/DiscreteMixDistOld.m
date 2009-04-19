classdef DiscreteMixDistOld < MixtureDist
% A mixture of products of Discrete Distributions, (i.e. Naive Bayes structure,
% but with hidden class labels). Each product of Discrete distributions is
% represented as a single vectorized DiscreteDist object. This class can be used
% to create a mixture of Bernoullis, simply fit it on binary data. 

    methods
        
        function model = DiscreteMixDistOld(varargin)
            [nmixtures,mixingWeights,distributions,model.transformer,model.nrestarts,model.verbose] = process_options(varargin,...
                'nmixtures',[],'mixingWeights',[],'distributions',[],'transformer',[],'nrestarts',model.nrestarts,'verbose',true);
            if(isempty(mixingWeights) && ~isempty(nmixtures))
                mixingWeights = DiscreteDist('T',normalize(ones(nmixtures,1)));
            end
            model.mixingWeights = mixingWeights;
            if(isempty(distributions)&&~isempty(model.mixingWeights))
                distributions = copy(DiscreteDist(),nstates(model.mixingWeights),1);
            end
            model.distributions = distributions;
        end
        
        function d = ndimensions(model)
           if(numel(model.distributions) > 0)
               d = ndistrib(model.distributions{1});
           else
               d = 0;
           end
        end     
    end
    
    methods(Access = 'protected')
        function model = initializeEM(model,data)
            n = size(data,1);
            nmixtures = numel(model.distributions);
            perm = randperm(n);
            batchSize = max(1,floor(n/nmixtures));
            for k=1:nmixtures
               start = (k-1)*batchSize+1;
               initdata = data(perm(start:start+batchSize-1),:);  
               model.distributions{k} = fit(model.distributions{k},'data',initdata);
            end 
        end 
    end
end