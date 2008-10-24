classdef gmm < handle
    %GMM Creates a Gaussian Mixture Model.
    %   obj = GMM(mu,sigma,pi) creates a Gaussian mixture model with values
    %   as follows:
    %
    %   mu      A K-by-D matrix specifying the mean of each of the
    %           D-dimensional, K components, i.e. mu(j,:) is the mean of
    %           component j. mu is stored in obj.mu. 
    %          
    %   sigma   Specifies the covariance matrix of each component. sigma is
    %           stored in obj.sigma and its dimensions must be one of the
    %           following:
    %
    %           * D-by-D-by-K array if there are no restrictions on the
    %             form of the covariance. In this case, sigma(:,:,J) is 
    %             the covariance matrix corresponding to component J.
    %
    %           * 1-by-D-by-K array if the covariance matrices are
    %             restricted to be diagonal, but not restricted to be same
    %             across components. In this case, sigma(:,:,J) contains
    %             the diagonal elements of the covariance of component J.
    %
    %           * D-by-D matrix if the covariance matrices are restricted
    %             to be the same across components, but not restricted to
    %             be diagonal.
    %
    %           * 1-by-D vector if the covariance matrices are restricted
    %             to be diagonal and the same across components. In this
    %             case, sigma contains the diagonal elements of the pooled
    %             covariance estimate.
    %
    %   pi       A 1-by-K vector specifying the mixing proportions of each
    %            component. pi is stored in obj.pi. The elements of pi
    %            automatically are normalized to sum to one. 
    %
    %  ADDITIONAL PROPERTIES:
    %
    %  obj.modelName default = 'Gaussian Mixture Model', A string naming
    %  the model.
    %
    %  obj.data, an n-by-d matrix storing the data used, or to be used,to
    %  fit the model. 
    %
    %  obj.sharedCov, default = false. True if the model specifies a single
    %  tied covariance matrix used by all components, else false. This
    %  parameter is used by fitting functions and in interpreting sigma. If
    %  mu and sigma are passed to the constructor, this value is inferred
    %  from their dimensions. This value can only be explicitly set if
    %  sigma has not yet been set. 
    %
    %  obj.covType, default = 'full'. One of two string values: 'full' |
    %  'diagonal' . This parameter is used by fitting functions and in
    %  interpreting sigma. If mu and sigma are passed to the constructor,
    %  this value is inferred from their dimensions.
    %
    %  obj.randomRestarts, default = 0. If this property is set,
    %  fitting functions fit the model this many times with randomly
    %  selected starting conditions and select the fit with the highest
    %  likelihood. Initial values of the parameters are ignored if this
    %  property is > 0. This parameter is ignored if obj.kmeansInit is
    %  true. 
    %   
    %  obj.kmeansInit, default = false. If this property is set to
    %  true, initial values and random restarts are not used in fitting,
    %  instead initial values are set via the kmeans algorithm. Values for
    %  pi are determined by the proportion of points in each cluster.
    %
    %  obj.nComponents, default = 0. The number of components in the model. Set
    %  automatically by either, mu, sigma, or pi. 
    %  
    %  obj.maxIterations, default = 500. Maximum number of iterations used
    %  in fitting this model. 
    % 
    %  obj.convergenceThreshold, default = 0.1. The fitting algorithm is
    %  deemed to have converged oncelog likelihood values from
    %  successive iterations differ by less than this amount. 
    %
    %  obj.crossValidate, default = false. If true, the properties
    %  nComponents,mu,sigma,pi are ignored during fitting and instead
    %  the number of components is determined via cross validation. The
    %  crossVaidationRange must also be specified - a model is fit for each
    %  value in this range and tested using a reserved portion of the data.
    %  Only the model corresponding to the highest likelihood is retained
    %  however, a pauseFunction() can be used to obtain interm results.
    %
    %  obj.crossValidationRange, default = 1:5. An array of integers
    %  specifying each value for nComponenents used in cross validation
    %  based fitting. 
    %
    %  obj.kfolds, default = 5. An integer specifying the number of folds
    %  to use if the crossValidation option is set. 
    %
    %  READ ONLY PROPERTIES
    %
    %  obj.nDimensions, The dimensionality of each component. Set
    %  automatically by either mu, sigma, data. This class
    %  enforces consistancy among these properties w.r.t. dimensionality.
    %
    %  obj.N, The number of data points in obj.data. Set automatically to
    %  the number of rows in obj.data.  
    % 
    %  obj.iterations, The current number of iterations performed so far in
    %  the fit. 
    %
    %  obj.converged, default = false. True iff the fitting algorithm has
    %  converged. 
    %
    %  METHODS
    % 
    %  obj.fit(), Fit the GMM using EM. The following properties affect
    %  this method, see their descriptions: mu,sigma,pi,data, sharedCov,
    %  covType,randomRestarts, kmeansInit, nComponents, maxIterations,
    %  convergenceThreshold, crossValidate, crossValidationRange.
    %
    %  [samples components] = obj.sample(n) Return n random samples from this
    %  distribution in an n-by-d matrix.
    % 
    %  P = obj.posterior(D) Return an n-by-k matrix of posterior
    %  probabilities where each P(j,k) is the posterior probability
    %  assigned by component k to data point j = D(i,:). If D is not
    %  specified, obj.data is used instead. 
    %
    %  ll = obj.logLikelihood(D), Return the expected complete data log likelihood of the
    %  specified data, D, given the model, (a single number), if D is not
    %  specified, obj.data is used instead. 
    %  
    %  obj.cluster() Partition the data into clusters, one cluster for
    %  every component of this model by hard assigning data points to
    %  components based on their posterior probabilities. Returns an N-by-1
    %  matrix of cluster numbers such that if obj.data(i,:) is in cluster
    %  k, then clusters(i) = k. 
    %
    %  p = obj.pdf(X), Return a vector of size N-by-1 corresponding to the
    %  value of the underlying pdf for each data point,(row), of input X. X
    %  is an N-by-D matrix. 
    %
    %  p = obj.pdfFactored(X), Return a vector of size N-by-K. Similar to
    %  pdf(), except pdf values for each component are calculated
    %  seperately and returned in the K columns of p. Component pdf values
    %  are scaled by their corresponding values for pi. pdfFactored is related
    %  to pdf by the following relation. Suppose a = pdf(X) and b =
    %  pdfFactored(X), then a = sum(b,2), that is, we can obtain a by
    %  summing b along each of its columns. 
    %
    %  obj.copy(), create and return an identical copy of this object.
    %  Since gmm inherits from handle, b = gmm; a = b; creates two handles
    %  for the same underlying object; use this method to create two
    %  distinct objects. 
    % 
    % obj.set('property',value,...) Set a number of properties at once by
    % specifying the property name in quotations followed by its value.
    %
    % obj.reset(), resets mu,sigma,pi and nComponents. If data is empty, it
    % also resets nDimensions. Does not affect any other property. 
    % 
    % Written by Matthew Dunham and Kevin Murphy
    % Version 1.0, March 30, 2008
    
    properties(GetAccess = 'public',SetAccess = 'public')
       mu = [];                      
       sigma = [];
       pi = [];
       modelName = 'Gaussian Mixture Model'; 
       data = [];
       sharedCov = false;
       covType = 'full';
       randomRestarts = 5;
       kmeansInit = false;
       nComponents = 0;
       maxIterations = 500;
       convergenceThreshold = 0.1;
       crossValidate = false;
       crossValidationRange = 1:5;
       kfolds = 5;
       regularizer = 0; % will be replaced by full MAP estimation
       verbosity = 1;
    end
  
    properties(GetAccess = 'public',SetAccess = 'private')
        nDimensions = 0;
        N = 0;
        iterations = 0;
        converged = false;
    end
    
    properties(GetAccess = 'private',SetAccess = 'private')
        prevLL = [];
        prevRnk = [];
        dirtyRnk = true;
        %Interwoven function handles
        IWupdate;
        IWrestart;
        IWcv;
        convergenceErrors = 0;
    end
    
    methods
       
        function obj = gmm(mu,sigma,pi)
            if(nargin > 0) 
                obj.mu = mu;
                obj.sigma = sigma;
                obj.pi = pi;
                if(isempty(obj.mu) || isempty(obj.sigma) || isempty(obj.pi))
                    obj.mu = [];obj.sigma = []; obj.pi = []; obj.nComponents = 0;
                    obj.nDimensions = 0; obj.covType = 'full'; obj.sharedCov = false;
                end
            end
            warning off all;
        end
        function set.mu(obj, mu)
            if(isempty(mu)),obj.mu = [];return;end;
            if(obj.consistant(mu,'mu'))
                obj.mu = []; 
                obj.nComponents = size(mu,1);
                obj.nDimensions = size(mu,2);
                obj.mu = mu;
                obj.dirtyRnk = true;
            end
        end
        function set.sigma(obj,sigma)
            if(isempty(sigma)),obj.sigma = [];return;end;
            if(obj.consistant(sigma,'sigma'))
                obj.sigma = [];
                obj.nDimensions = size(sigma,2);
                if(size(sigma,1) == 1)
                   obj.covType = 'diagonal';
                else obj.covType = 'full';
                end
                obj.sharedCov = (size(sigma,3) == 1) && (obj.nComponents > 1);
                obj.sigma = sigma;
                obj.dirtyRnk = true;
            end
        end
        function set.pi(obj,pi)
            if(isempty(pi)),obj.pi = [];return;end;
            if(obj.consistant(pi,'pi'))
                obj.pi = [];
                obj.nComponents = size(pi,2);
                obj.pi = [pi./sum(pi)];
                obj.dirtyRnk = true;
            end
        end
        function set.data(obj,data)
            if(isempty(data)),obj.data = [];return;end;
            if(obj.consistant(data,'data'))
                obj.data = [];
                obj.nDimensions = size(data,2);
                obj.N = size(data,1);
                obj.data = data;
                obj.converged = false;
                obj.prevRnk = [];
            end
        end
        function set.covType(obj,covType)
            if(ischar(covType)),covType = lower(covType);else gmm.typeError;return;end;
            if(isempty(obj.sigma))
                if(ismember(covType,{'full','diagonal'}))
                    obj.covType = covType;
                else
                    gmm.covTypeError;return;
                end
            else
                if(strcmp(covType,'full') && strcmp(obj.covType, 'diagonal'))
                    obj.covType = covType;
                    sigma = zeros(obj.nDimensions,obj.nDimensions,size(obj.sigma,3));
                    for k=1:size(obj.sigma,3)
                        sigma = diag(obj.sigma(:,:,k));
                    end
                    obj.sigma = sigma;
                end 
            end
            obj.dirtyRnk = true;
        end
        function set.sharedCov(obj,sharedCov)
            if(~islogical(sharedCov) && (sharedCov~=0) && (sharedCov~=1)), gmm.typeError;return;end;
            if(isempty(obj.sigma))
                obj.sharedCov = sharedCov;
            else
                gmm.propertyError('sharedCov');
            end
            obj.dirtyRnk = true;
        end
        function set.nComponents(obj,nComponents)
            if(obj.consistant(nComponents,'nComponents'))
                obj.nComponents = nComponents;
                obj.dirtyRnk = true;
            end
        end
        function set.modelName(obj,modelName)
           if(~ischar(modelName))
               gmm.typeError('modelName');return;
           end
           obj.modelName = modelName;
        end
        function set.randomRestarts(obj,randomRestarts)
             if(~isscalar(randomRestarts) || randomRestarts < 0)
                gmm.typeError('randomRestarts');return;
             end
             if(obj.kmeansInit && randomRestarts > 0)
                 gmm.kmeansInitWarning;
             end
             if((randomRestarts > 0) && (~isempty(obj.mu) || ~isempty(obj.sigma) || ~isempty(obj.pi)))
                 gmm.randomRestartWarning;
             end
            obj.randomRestarts = randomRestarts;
        end
        function set.kmeansInit(obj,kmeansInit)
            if(~islogical(kmeansInit) && (kmeansInit ~=0) &&(kmeansInit ~=1))
                gmm.typeError('kmeansInit');return;
            end
            if(obj.randomRestarts > 0 && kmeansInit)
                gmm.kmeansInitWarning;
            end
            obj.kmeansInit = kmeansInit;
        end
        function set.maxIterations(obj,maxIterations)
           if(~isscalar(maxIterations) || ~isnumeric(maxIterations) || maxIterations < 1)
               gmm.typeError('maxIterations');return;
           end
           obj.maxIterations = maxIterations;  
        end
        function set.convergenceThreshold(obj,convergenceThreshold)
            if(~isscalar(convergenceThreshold) || ~isnumeric(convergenceThreshold) || convergenceThreshold < 1e-30)
               gmm.typeError('convergenceThreshold');return;
           end
           obj.convergenceThreshold = convergenceThreshold; 
        end
        function set.crossValidate(obj,crossValidate)
             if(~islogical(crossValidate) && (crossValidate ~=0) &&(crossValidate ~=1))
                gmm.typeError('crossValidate');return;
            end
            obj.crossValidate = crossValidate;
            if(crossValidate && (~isempty(obj.mu) || ~isempty(obj.sigma) || ~isempty(obj.pi) || (obj.nComponents > 0)))
                gmm.crossValidateWarning
            end
                
        end
        function set.crossValidationRange(obj,crossValidationRange)
            if(isempty(crossValidationRange) || ~isnumeric(crossValidationRange) || ~all(crossValidationRange(:) > 0))
                gmm.typeError('crossValidationRange');return;
            end
            obj.crossValidationRange = crossValidationRange;
        end
        function set.kfolds(obj,kfolds)
            if(isscalar(kfolds) && (kfolds > 1))
                obj.kfolds = kfolds;
            else
                gmm.typeError('kfolds');
            end
        end
        function set(obj,varargin)
            full ='full';diagonal = 'diagonal';
            i = 1;
            while(i < length(varargin)) 
                eval(['obj.',varargin{i},' = ',' [',mat2str(varargin{i+1}),']']);
                i = i+2;
            end
        end
        
        function fit(obj)
        % Fit via EM according to this object's current properties
            if(isempty(obj.data)), gmm.dataError(); return; end
            if(obj.nComponents == 0)
                if(~obj.crossValidate),gmm.cvMessage;end
                obj.crossValidate = true;
            end
                if(obj.crossValidate)
                    obj.fitCV;
                elseif(obj.randomRestarts > 0)
                    obj.fitRR;
                else
                    obj.fitVanilla
                end
            disp(obj);
        end
        function reset(obj)
           obj.mu = [];
           obj.sigma = [];
           obj.pi = [];
           obj.nComponents = 0;
           if(isempty(obj.data))
               obj.nDimensions = 0;
           end
        end   
        function [samples components] = sample(obj,n)
        % Randomly sample n-times from this distribution
            if(isempty(obj.mu) || isempty(obj.sigma) || isempty(obj.pi))
                gmm.sampleError;
                samples = 0; components = 0;
                return;
            end
            components = sampleDiscrete(obj.pi,n,1);
            if(obj.sharedCov || (obj.nComponents == 1))
               samples = mvnrandom(obj.mu(components,:),obj.sigma); 
            else
               samples = mvnrandom(obj.mu(components,:),obj.sigma(:,:,components));
            end
        end
        function p = posterior(obj,D)
        % Return posterior probabilities of the data points
            if(isempty(obj.mu) || isempty(obj.sigma) || isempty(obj.pi))
              gmm.emptyPropertyError;
              p = [0];
              return;
           end
            if(nargin < 2), D = obj.data;end
            if(isempty(D)), gmm.dataError; return;end
            p = normalize(obj.calcPost(D),2);
        end      
        function llik = logLikelihood(obj,D)
        % Return the expected complete data log likelihood of the specified data given
        % this model. If data is not specified use obj.data
            if(nargin < 2)
                D = obj.data;
                if(obj.converged),llik = obj.prevLL; return; end
            end
            if(isempty(obj.mu) || isempty(obj.sigma) || isempty(obj.pi))
              gmm.emptyPropertyError;
              llik = 1;
              return;
           end
            
            if(isempty(D)), gmm.dataError; return; end
            
            Rnk = obj.calcPost(D);
            llik = sum(log(sum(Rnk,2)),1);
        end
        function clusters = cluster(obj)
        % Hard assign data points to components
            post = obj.posterior;
            [vals clusters] = max(post,[],2);
        end       
        function p = pdf(obj,X)
        % Evaluate the pdf at every point,(row) in X
            if(isempty(obj.mu) || isempty(obj.sigma) || isempty(obj.pi))
              gmm.emptyPropertyError;
              p = zeros(size(X,1),1);
              return;
           end
            f = obj.makeFunction();
            p = f(X);
        end       
        function p = pdfFactored(obj,X)
             if(isempty(obj.mu) || isempty(obj.sigma) || isempty(obj.pi))
                gmm.emptyPropertyError;
                p = zeros(size(X,1),obj.nComponents); 
                return;
             end
             p = zeros(size(X,1),obj.nComponents);
             if(obj.sharedCov || (obj.nComponents == 1))
                 if(strcmp(obj.covType,'full'))
                     for i=1:obj.nComponents
                        p(:,i) = obj.pi(i)*mvnormpdf(X,obj.mu(i,:),obj.sigma);                        
                     end
                 else
                     for i=1:obj.nComponents
                         p(:,i) = obj.pi(i)*mvnormpdf(X,obj.mu(i,:),diag(obj.sigma));                         
                    end 
                 end
             else
                 if(strcmp(obj.covType,'full'))
                     for i=1:obj.nComponents
                        p(:,i) = obj.pi(i)*mvnormpdf(X,obj.mu(i,:),obj.sigma(:,:,i));                         
                     end
                 else
                     for i=1:obj.nComponents
                         p(:,i) = obj.pi(i)*mvnormpdf(X,obj.mu(i,:),diag(obj.sigma(:,:,i)));                        
                     end
                 end
             end 
        end   
        function obj2 = copy(obj1)
        % Construct and return a full copy of the object    
            obj2 = gmm(obj1.mu,obj1.sigma,obj1.pi);
            obj2.modelName = obj1.modelName;
            obj2.set('data',obj1.data,'covType',obj1.covType,...
                'randomRestarts',obj1.randomRestarts,...
                'kmeansInit',obj1.kmeansInit,'maxIterations',obj1.maxIterations,...
                'convergenceThreshold',obj1.convergenceThreshold,'crossValidate',...
                 obj1.crossValidate,'crossValidationRange',obj1.crossValidationRange)
        end
        function interweave(obj,handle,location)
            if(~isa(handle,'function_handle'))
                gmm.notAfunctionError;
                return;
            end
            if(strcmp(location,'update'))
                obj.IWupdate = handle;
            elseif(strcmp(location,'restart'))
                obj.IWrestart = handle;
            elseif(strcmp(location,'cv'))
                obj.IWcv = handle;
            else
                gmm.invalidLocation;
            end
        end
        function clearInterweaver(obj,location)
             if(strcmp(location,'update'))
                obj.IWupdate = [];
            elseif(strcmp(location,'restart'))
                obj.IWrestart = [];
            elseif(strcmp(location,'cv'))
                obj.IWcv = [];
            else
                gmm.invalidLocation;
            end
        end
        function properties(obj)
        % Display the properties of this class
           fprintf(['\nmodelName: ',obj.modelName,'\nnDimensions: ',num2str(obj.nDimensions),'\nnComponents: ',num2str(obj.nComponents),'\nN: ',num2str(obj.N),'\nmu: ...\nsigma: ...\npi: ...\ndata: ...\nsharedCov: ',num2str(obj.sharedCov),'\ncovType: ',num2str(obj.covType),'\nrandomRestarts: ',num2str(obj.randomRestarts),'\nkmeansInit: ',num2str(obj.kmeansInit),'\nmaxIterations: ', num2str(obj.maxIterations),'\nconvergenceThreshod: ',num2str(obj.convergenceThreshold),'\ncrossValidate: ',num2str(obj.crossValidate),'\ncrossValidationRange: ',mat2str(obj.crossValidationRange),'\nkfolds:',num2str(obj.kfolds),'\niterations: ',num2str(obj.iterations),'\nconverged: ',num2str(obj.converged),'\n']);
        end
        function disp(obj)
       % Display the name, number of components and dimensionality of
       % the model 
            fprintf([obj.modelName,' (',num2str(obj.nComponents),' components) (',num2str(obj.nDimensions),' dimensions)\n']);
        end      
        function methods(obj)
        %Display the methods of this class
          fprintf('gmm\nfit\nsample\nposterior\nlogLikelihood\ncluster\npdf\npdfFactored\ncopy\ninterweave\nclearInterweaver\nmethods\nproperties\nset\nreset\n');
        end
       
    end % end of public methods
    
    methods(Access = 'private')
        function bool = consistant(obj,property,propertyName)
        % Check that dimensions and components are consistant
            bool = true;
            if(isempty(property)), return;end;
            
            if(strcmp(propertyName,'mu'))
                bool = bool && isnumeric(property);
                if(~bool),gmm.typeError('mu');return;end;
                if(obj.nDimensions > 0)
                   bool = bool && (size(property,2) == obj.nDimensions); 
                   if(~bool),gmm.dimError('mu');return;end;
                end
                if(obj.nComponents > 0)
                    bool = bool && ((size(property,1) == obj.nComponents));
                    if(~bool),gmm.compError('mu');return;end;
                end
                
            elseif(strcmp(propertyName,'sigma'))
                bool = bool && isnumeric(property);
                if(obj.nDimensions > 0)
                   bool = bool && size(property,2) == obj.nDimensions;
                  if(~bool),gmm.dimError('sigma');return;end;
                end
                bool = bool && ((size(property,1) == size(property,2)) || (size(property,1) == 1));
                if(~bool),gmm.dimError('sigma');return;end;
                
                if(obj.nComponents > 0)
                   bool = bool && ((size(property,3) == obj.nComponents)|| size(property,3) == 1);
                   if(~bool),gmm.compError('sigma');return;end;
                end
                
            elseif(strcmp(propertyName, 'pi'))
                bool = bool && isnumeric(property) && all(property(:) >=0) && (sum(property(:)) > 0);
                if(~bool),gmm.typeError('pi');return;end;
                bool = bool && (size(property,1) == 1);
                if(~bool),gmm.dimError('pi');return;end;
                if(obj.nComponents > 0)
                   bool = bool && (size(property,2) == obj.nComponents);
                   if(~bool),gmm.compError('pi');return;end;
                end
            
            elseif(strcmp(propertyName,'data'))
                bool = bool && isnumeric(property);
                if(~bool),gmm.typeError('data');return;end;
                if(obj.nDimensions > 0)
                    bool = bool && size(property,2) == obj.nDimensions;
                    if(~bool),gmm.dimError('data');return;end;
                end
            elseif(strcmp(propertyName,'nComponents'))
                bool = bool && isscalar(property) && property >= 0;
                bool = bool && (isempty(obj.mu) || size(obj.mu,1) == property)...
                            && (isempty(obj.sigma) || size(obj.sigma,3) == property || size(obj.sigma,3) == 1)...
                            && (isempty(obj.pi) || size(obj.pi,2) == property);
                           
                if(~bool), gmm.compError('nComponents'); end
            end   
        end % end of consistant()      
        function f = makeFunction(obj)
        %Construct a function handle out of this distribution
           if(isempty(obj.mu) || isempty(obj.sigma) || isempty(obj.pi))
              gmm.emptyPropertyError;
               f = @(X)0;
              return;
           end
        
            f = @(X)0;
             if(obj.sharedCov || (obj.nComponents == 1))
                 if(strcmp(obj.covType,'full'))
                     for i=1:obj.nComponents
                        f = @(X)f(X) + obj.pi(i)*mvnormpdf(X,obj.mu(i,:),obj.sigma); 
                     end
                 else
                     for i=1:obj.nComponents
                        f = @(X)f(X) + obj.pi(i)*mvnormpdf(X,obj.mu(i,:),diag(obj.sigma)); 
                    end 
                 end
             else
                 if(strcmp(obj.covType,'full'))
                     for i=1:obj.nComponents
                        f = @(X)f(X) + obj.pi(i)*mvnormpdf(X,obj.mu(i,:),obj.sigma(:,:,i)); 
                     end
                 else
                     for i=1:obj.nComponents
                        f = @(X)f(X) + obj.pi(i)*mvnormpdf(X,obj.mu(i,:),diag(obj.sigma(:,:,i))); 
                     end
                 end
             end 
        end   
        function fitCV(obj)
        %Fit via EM but pick nComponents based on cross validation
            foldSize = floor(obj.N / obj.kfolds);
            perm = randperm(obj.N);
            extra = perm(obj.kfolds*foldSize + 1:end);
            perm = perm(1:obj.kfolds*foldSize);
            fold = reshape(perm,foldSize,obj.kfolds);
            fullData = obj.data;
            testloglik = zeros(length(obj.crossValidationRange),1);
            for c =1:length(obj.crossValidationRange)
                obj.reset();
                obj.nComponents = obj.crossValidationRange(c);
                for k=1:obj.kfolds
                    testNdx = fold(:,k);
                    trainNdx = [setdiff(perm,testNdx),extra];
                    testSet = fullData(testNdx,:);
                    obj.data = fullData(trainNdx,:);
                    if(obj.randomRestarts > 0)
                        obj.fitRR();
                    else
                        obj.fitVanilla();
                    end
                    testloglik(c) = testloglik(c) + obj.logLikelihood(testSet);
                end
                testloglik(c) = testloglik(c) / obj.kfolds;
                if(obj.verbosity == 1)
                   fprintf(['\nCV nComponents: ',num2str(obj.nComponents),'\nmean test ll: ',num2str(obj.prevLL),'\n']);  
                end
                if(~isempty(obj.IWcv))
                    obj.IWcv();
                end
            end
            if(any(isnan(testloglik)))
                gmm.fitError;
                assert(false); 
            end
            oneSTD = std(testloglik(~isnan(testloglik))); 
            maxVal = max(testloglik(~isnan(testloglik)));
            candidates = (~isnan(testloglik)) & (testloglik <= (maxVal + oneSTD)) & (testloglik >= (maxVal - oneSTD));
            best = min(obj.crossValidationRange(candidates));
            obj.reset();
            obj.data = fullData;
            obj.nComponents = best;
            
           if(obj.randomRestarts > 0)
                obj.fitRR();
           else
                obj.fitVanilla();
           end
        end
        function fitRR(obj)
        %Fit via EM using random restarts
              bestLL = [];
              for r = 1: obj.randomRestarts + 1
                  obj.fitVanilla();
                  if((isempty(bestLL) || obj.prevLL > bestLL))
                      bestLL = obj.prevLL;
                      bestMU = obj.mu;
                      bestSIGMA = obj.sigma;
                      bestPI = obj.pi;
                  end
                  if(~isempty(obj.IWrestart))
                      obj.IWrestart();
                  end
              end
                obj.reset();
                obj.mu = bestMU;
                obj.sigma = bestSIGMA;
                obj.pi = bestPI;
                if(obj.verbosity == 2)
                    fprintf(['\nRR nComponents: ',num2str(obj.nComponents),'\nTEST ll: ',num2str(bestLL),'\n']); 
                end
        end
        function fitVanilla(obj)
        %Regular one-pass EM
           
                obj.initializeFit();
                while(~obj.converged && (obj.iterations < obj.maxIterations))
                    obj.EMupdate();
                    obj.iterations = obj.iterations + 1;
                    currentLL = obj.logLikelihood();
                    
                    if(abs(obj.prevLL - currentLL) < obj.convergenceThreshold)
                        obj.converged = true;
                    end
                    obj.prevLL = currentLL;
                end
                if(obj.iterations >= obj.maxIterations)
                    if(obj.convergenceErrors == 0)
                        fprintf(['\nWARNING: did not converge after ',num2str(obj.maxIterations),' iterations.\n']);
                        obj.convergenceErrors = obj.convergenceErrors + 1;
                    else
                        obj.convergenceErrors = obj.convergenceErrors + 1;
                    end
                end
        end
        function initializeFit(obj)
            setMu = false; setSigma = false; setPi = false;
            if(obj.kmeansInit())
                obj.mu = kmeansKPM(obj.data,obj.nComponents,obj.maxIterations,obj.convergenceThreshold);
            elseif(obj.randomRestarts > 0)
                setMu = true; setSigma = true; setPi = true;
            end
            if(isempty(obj.mu)),setMu = true;end
            if(isempty(obj.sigma)),setSigma = true; end
            if(isempty(obj.pi)),setPi = true; end
            if(setPi), obj.pi = ones(1,obj.nComponents);end
            if(setMu)
                ndx = repmat(randperm(obj.N),1,ceil(obj.nComponents / obj.N));
                obj.mu = obj.data(ndx(1:obj.nComponents),:)+rand(obj.nComponents,obj.nDimensions);
            end
            
            if(setSigma)
                dataCov = cov(obj.data);
                mx = max(dataCov(:));
               if(strcmp(obj.covType,'full'))
                    sigma = dataCov ./ mx;
               elseif(strcmp(obj.covType,'diagonal'))
                    sigma = (diag(dataCov) ./ mx)';
               else
                   m = mean(diag(dataCov))./mx;
                   sigma = repmat(m,1,obj.nDimensions);
               end
 
               if(obj.sharedCov || (obj.nComponents == 1))
                   obj.sigma = sigma;
               else
                   obj.sigma = repmat(sigma,[1,1,obj.nComponents]);
               end
            end
        end
        function Rnk = calcPost(obj,D)
        %Calculate unnormalized posterior probabiliites. See obj.posterior
        %or obj.logLikelihood
            if(nargin < 2)
                if(~obj.dirtyRnk && ~isempty(obj.prevRnk)), Rnk = obj.prevRnk; return; end
                D = obj.data;
            end
                
            N = size(D,1);
            K = obj.nComponents;
            Rnk = zeros(N,K);
            
            if(K == 1)
                Rnk(:,1) = obj.pi*mvnormpdf(D,obj.mu(1,:),obj.sigma);
            elseif(obj.sharedCov)
                for k=1:K
                    Rnk(:,k) = obj.pi(k)*mvnormpdf(D,obj.mu(k,:),obj.sigma);
                end
            else
                for k=1:K
                    Rnk(:,k) = obj.pi(k)*mvnormpdf(D,obj.mu(k,:),obj.sigma(:,:,k));
                end
            end
            obj.prevRnk = Rnk;
            obj.dirtyRnk = false; 
        end
        function EMupdate(obj)
        % Perform a single iteration of vanilla expectation maximization.
        % Assumes that initialization has already been done and uses
        % current values for mu,sigma,pi as old values. The following
        % properties affect this method: covType,sharedCov. It is up to
        % calling function to check for convergence, update number of
        % iterations, calculate log likelihood, etc.  
            %E step 
            Rnk =  normalize(obj.calcPost(obj.data),2); 
            if(any(isnan(Rnk)))
                gmm.posDefError;
                assert(false);
            end
            %M step
            Rk = sum(Rnk,1); 
            obj.mu = Rnk'*obj.data ./ repmat(Rk',1,obj.nDimensions);
            if(strcmp(obj.covType,'full'))
                sigma = zeros(obj.nDimensions,obj.nDimensions,obj.nComponents);
                for k = 1:obj.nComponents
                    d = obj.data - repmat(obj.mu(k,:),obj.N,1);
                    sigma(:,:,k) = ((repmat(Rnk(:,k),1,obj.nDimensions).*d)'*d)./Rk(k) + diag(obj.regularizer*ones(1,obj.nDimensions));
                end
            else
                sigma = zeros(1,obj.nDimensions,obj.nComponents);
                for k = 1:obj.nComponents
                    d = (obj.data - repmat(obj.mu(k,:),obj.N,1)).^2;
                    sigma(1,:,k) = (Rnk(:,k)'*d)./Rk(k) + obj.regularizer*ones(1,obj.nDimensions);
                end
            end
            if(obj.sharedCov)
                obj.sigma = sum(sigma,3)./obj.nComponents;
            else
                obj.sigma = sigma;
            end
            obj.pi = (Rk ./ obj.N);  
            if(obj.verbosity == 3)
                   fprintf(['\nnComponents: ',num2str(obj.nComponents),'\nll: ',num2str(obj.prevLL),'\n']); 
            end
        
            if(~isempty(obj.IWupdate))
                obj.IWupdate();
            end
        end
    end % end of private methods
    
    
    
    methods(Access = 'private', Static = true)  
        function dimError(property)
           fprintf(['\nERROR setting ', property, ' : Setting this/these value(s) would result\nin inconsistant dimensions. No change has been made.\n\n']);
        end
        function compError(property)
           fprintf(['\nERROR setting ', property, ' : Setting this/these value(s) would result\nin an inconsistant number of components. No change has been made.\n\n']);
        end       
        function typeError(property)
            fprintf(['\nERROR setting ', property,' : Invalid Type - no change has been made.\n']);
          
        end
        function covTypeError
           fprintf('\nERROR: covType can only be set to one of\n"full" | "diagonal".\n'); 
        end
        function propertyError(property)
           fprintf(['\nERROR: ', property,' can only be explicitly\nset if the property to which it refers is [].\nTry setting the latter to [] first.\n\n']); 
        end
        function posDefError
           fprintf('\nERROR: One or more covariance matricies is not positive definite.\n');
        end
        function sampleError
           fprintf('\nERROR: mu,sigma,and pi must first be set before you can sample.\n'); 
        end       
        function dataError
            fprintf('\nERROR: data has not been set.\n'); 
        end
        function kmeansInitWarning
            fprintf('\nWARNING: Initial values for mu, sigma, pi, and the random restart setting\nare ignored when kmeansInit is set to true.\n');
        end
        function randomRestartWarning
           fprintf('\nWARNING: Intial values for mu,sigma, and pi are ignored\nwhen random restarts is set to a value > 0.\n'); 
        end
        function crossValidateWarning
           fprintf('\nWARNING: Initial values for mu, sigma, pi, and nComponents\nare ignored when crossValidate is set to true.\n'); 
        end
        function cvMessage
           fprintf('nComponents not set - performing cross validation...\n');
        end
        function fitError
           fprintf('\nERROR: One or more errors occurred during fitting.\n');  
        end
        function invalidLocation
            fprintf('\nERROR: This is an invalid location.\n');
        end
        function notAfunctionError
           fprintf('\nERROR: This is not a valid function handle.\n'); 
        end
        function emptyPropertyError
           fprintf('\nERROR: one or more of mu, sigma, pi has not been set\n'); 
        end
                   
    end % end of private static methods
   
    
       
end % end of class definition


function M = sampleDiscrete(prob, r, c)
% SAMPLEDISCRETE Like the built in 'rand', except we draw from a non-uniform discrete distrib.
% M = sample_discrete(prob, r, c)
%
% Example: sampleDiscrete([0.8 0.2], 1, 10) generates a row vector of 10 random integers from {1,2},
% where the prob. of being 1 is 0.8 and the prob of being 2 is 0.2.

    n = length(prob);
    if nargin == 1
        r = 1; c = 1;
    elseif nargin == 2
        c = r;
    end
    R = rand(r, c);
    M = ones(r, c);
    cumprob = cumsum(prob(:));
    if n < r*c
        for i = 1:n-1
            M = M + (R > cumprob(i));
        end
    else
        % loop over the smaller index - can be much faster if length(prob) >> r*c
        cumprob2 = cumprob(1:end-1);
        for i=1:r
            for j=1:c
                M(i,j) = sum(R(i,j) > cumprob2)+1;
            end
        end
    end
end

function p = mvnormpdf(X,mu,sigma)
% Multivariate normal pdf
    d = size(mu,2);
    n = size(X,1);
    if((size(sigma,1) == 1) && (d > 1))
        sigma = spdiags(sigma',0,zeros(d,d));
    end
    Xs = X - repmat(mu,n,1);
    qd = sum((Xs*(chol(inv(sigma)))').^2,2);
    numer = exp((-1/2)*qd);
    denom = ((2*pi)^(d/2))*((det(sigma))^(1/2));
    p = numer ./ denom; 
end

function r = mvnrandom(mu,sigma)
% Sample randomly from a multivariate normal distribution. mu is a n-by-d
% matrix and sigma is one of four possible sizes: 1-by-d, d-by-d,
% 1-by-d-by-n, d-by-d-by-n. 
    d = size(mu,2);
    n = size(mu,1);
    dim1 = size(sigma,1);
    dim2 = size(sigma,2);
    dim3 = size(sigma,3);
    
    Z = randn(d,n);
    
    %diagonal, shared
    if((dim1 == 1) && (dim2 > 1) && (dim3 == 1))
        A = spdiags(sqrt(sigma)',0,zeros(d,d));
        r = (mu' + A*Z)';
    
    %full, shared
    elseif((dim1 == dim2) && (dim3 == 1))
        A = (chol(sigma))';
        r = (mu' + A*Z)';
    
    %diagonal, not shared
    elseif((dim1 == 1) && (dim2 > 1) && (dim3 > 1))
        r = zeros(n,d);
        for i=1:n
           A = spdiags(sqrt(sigma(:,:,i))',0,zeros(d,d));
           r(i,:) = (mu(i,:)' + A*Z(:,i))';
        end
    
    % full, not shared
    elseif((dim1 == dim2) && (dim3 > 1))
        r = zeros(n,d);
        for i=1:n
            A = (chol(sigma(:,:,i)))';
            r(i,:) = (mu(i,:)' + A*Z(:,i))';
        end
    end
end

function mu = kmeansKPM(data, K, maxIter, thresh)

[N D] = size(data);
% initialization by picking random pixels
% mu(k,:) = k'th center
perm = randperm(N);
mu = data(perm(1:K),:);

converged = 0;
iter = 1;
while ~converged && (iter < maxIter)
  newmu = zeros(K,D);
  % dist(i,k) = squared distance from pixel i to center k
  dist = sqdist(data', mu');
  [junk, assign] = min(dist,[],2);
  for k=1:K
    newmu(k,:) = mean(data(assign==k,:), 1);
  end
  delta = abs(newmu(:) - mu(:)); 
  if max(delta./abs(mu(:))) < thresh
    converged = 1;
  end
  mu = newmu;
  iter = iter + 1;
end

function m = sqdist(p, q, A)
% SQDIST      Squared Euclidean or Mahalanobis distance.
% SQDIST(p,q)   returns m(i,j) = (p(:,i) - q(:,j))'*(p(:,i) - q(:,j)).
% SQDIST(p,q,A) returns m(i,j) = (p(:,i) - q(:,j))'*A*(p(:,i) - q(:,j)).

%  From Tom Minka's lightspeed toolbox

[d, pn] = size(p);
[d, qn] = size(q);

if nargin == 2
  pmag = sum(p .* p, 1);
  qmag = sum(q .* q, 1);
  m = repmat(qmag, pn, 1) + repmat(pmag', 1, qn) - 2*p'*q;
  %m = ones(pn,1)*qmag + pmag'*ones(1,qn) - 2*p'*q;
else
  if isempty(A) | isempty(p)
    error('sqdist: empty matrices');
  end
  Ap = A*p;
  Aq = A*q;
  pmag = sum(p .* Ap, 1);
  qmag = sum(q .* Aq, 1);
  m = repmat(qmag, pn, 1) + repmat(pmag', 1, qn) - 2*p'*Aq;
end


end

end
