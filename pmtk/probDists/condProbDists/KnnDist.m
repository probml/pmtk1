classdef KnnDist < CondProbDist
% A probabilistic K-nearest neighbour classifier - probabilistic in the sense
% that we return a distribution over class labels. 



    properties
        K;              % the number of neighbours to consider
        examples;       % a set of example points nexamples-by-ndimensions
        examplesSOS;    % sum(examples.^2,2)
        labels;         % labels corresponding to the above examples
        nclasses;       % the number of classes
        support;        % the support of the labels
        transformer;    % a data transformer object, e.g. PcaTransformer
        localKernel;    % e.g. Gaussian Kernel
        distanceFcn;    % the global metric, e.g. Euclidean distance: either the string 'sqdist' or a function handle
        useSoftMax      % if true, counts are smoothed using the softmax function, with specified inv temp beta
        beta;           % inverse temperate used in softmax smoothing S(y,beta) = normalize(exp(beta*y)
    end
    
    methods
       
        function obj = KnnDist(varargin)
            [obj.K,obj.transformer,obj.localKernel,obj.distanceFcn,obj.useSoftMax,obj.beta] =...
                process_options(varargin,...
                'K'             , 1              ,...
                'transformer'   , []             ,...
                'localKernel'   , []             ,...
                'distanceFcn'   , 'sqdist'       ,...
                'useSoftMax'    , true           ,...
                'beta'          , 1              );
            
                if ischar(obj.localKernel)
                   obj = setLocalKernel(obj,obj.localKernel);
                end
        end
        
        function obj = fit(obj,examples,labels)
           if(~isempty(obj.transformer))
               [examples,obj.transformer] = train(obj.transformer,examples);
           end
           obj.examples = examples;
           obj.examplesSOS = sum(examples.^2,2);
           [obj.labels,obj.support] = canonizeLabels(labels);
           obj.nclasses = numel(obj.support);
        end
        
        function pred = predict(obj,data)
            if(~isempty(obj.transformer))
                data = test(obj.transformer,data);
            end
            ntest = size(data,1);
            probs = zeros(ntest,obj.nclasses);
            batch = largestBatch(obj,1,ntest);
            wbar = waitbar(0,sprintf('%d of %d classified',0,ntest));
            tic;
            while ~isempty(batch)
                probs(batch,:) = predictHelper(obj,data(batch,:));
                t = toc; waitbar(batch(end)/ntest,wbar,sprintf('%d of %d Classified\nElapsed Time: %.2f seconds',batch(end),ntest,t));
                batch = largestBatch(obj,batch(end)+1,ntest);
            end
            close(wbar);
            pred = DiscreteDist('mu',probs','support',obj.support);
        end
        
    end
    
    methods(Access = 'protected')
        
        
        function probs = predictHelper(obj,data)
            if isequal(obj.distanceFcn,'sqdist')
                dst = sqDistance(data,obj.examples,sum(data.^2,2),obj.examplesSOS);
            else
                dst = obj.distanceFcn(data,obj.examples);
            end
            [sortedDst,kNearest] = minK(dst,obj.K);
            if ~isempty(obj.localKernel)
                weights = obj.localKernel(data,obj.examples,sortedDst,kNearest);
                counts = zeros(size(data,1),obj.nclasses);
                for j=1:obj.nclasses
                    counts(:,j) = counts(:,j) + sum((obj.labels(kNearest) == j).*weights,2);
                end
            else
                counts = histc(obj.labels(kNearest),1:obj.nclasses,2);
            end
            if(obj.useSoftMax)
                probs = normalize(exp(counts*obj.beta),2);
            else
                probs = normalize(counts,2);
            end 
        end
        
        function batch = largestBatch(obj,start,ntest)
           
            if start > ntest
                batch = []; return;
            end
            prop = 0.20; % proportion of largest possible array size to use
            batchSize = ceil( (prop*subd(memory,'MaxPossibleArrayBytes')/8) /size(obj.examples,1));
            batch = start:min(start+batchSize-1,ntest);
        end
        
        function obj = setLocalKernel(obj,kernelName)
            
           switch lower(kernelName)
              
               case 'epanechnikov'
                    obj.localKernel = @epanechnikovKernel;
               case 'tricube'
                    obj.localKernel = @tricubeKernel;
               case 'gaussian'
                    obj.localKernel = @gaussianKernel;
                    
               otherwise
                   error('The %s local kernel has not been implemented, please pass in your own function instead',kernelName);
               
                   
           end
            function weights = gaussianKernel(testData,examples,sqdist,knearest)
                bandwidth = repmatC(sqrt(max(sqdist,[],2))+eps,1,size(knearest,2));
                weights = normalize(eps + (1./((2*pi)*bandwidth)).*exp(-(sqdist./(2*bandwidth.^2))),2);
            end
            
            function weights = tricubeKernel(testData,examples,sqdist,knearest)
                bandwidth = repmatC(sqrt(max(sqdist,[],2))+eps,1,size(knearest,2));
                weights = sqrt(sqdist)./bandwidth;
                in = abs(weights) <= 1;
                weights(in)  = (1-weights(in).^3).^3;
                weights(not(in)) = 0;
                weights = normalize(eps + weights,2);
            end
            
            function weights = epanechnikovKernel(testData,examples,sqdist,knearest)
                bandwidth = repmatC(sqrt(max(sqdist,[],2))+eps,1,size(knearest,2));
                weights = sqrt(sqdist)./bandwidth;
                in = abs(weights) <= 1;
                weights(in)  = (3/4)*(1-weights(in).^2);
                weights(not(in)) = 0;
                weights = normalize(eps + weights,2);
            end
                   
                   
           
            
            
        end
        
        
    end
    
    methods(Static = true)
        
        function testClass()
            profile on;
            load mnistAll;
            if 1
                trainndx = 1:60000; testndx =  1:1000;
            else
                trainndx = 1:10000;
                testndx =  1:1000;
            end
            ntrain = length(trainndx);
            ntest = length(testndx);
            Xtrain = double(reshape(mnist.train_images(:,:,trainndx),28*28,ntrain)');
            Xtest  = double(reshape(mnist.test_images(:,:,testndx),28*28,ntest)');
            if 1
                Xtrain = sparse(Xtrain);
                Xtest  = sparse(Xtest);
            end
            
            ytrain = (mnist.train_labels(trainndx));
            ytest  = (mnist.test_labels(testndx));
            clear mnist;
            model = KnnDist('K',3,'localKernel','gaussian');
            model = fit(model,Xtrain,ytrain);
            clear Xtrain ytrain
            pred = predict(model,Xtest);
            err = mean(ytest ~= mode(pred))
            profile viewer
            
            
        end
        
        
    end
    
    
    
    
    
end