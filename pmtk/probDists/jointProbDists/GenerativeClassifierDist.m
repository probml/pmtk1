classdef GenerativeClassifierDist < ParamDist
    
    properties
        classConditionals;      % a cell array of distributions 1 per class
        classPrior;             % Discrete or Discrete_Dirichlet 
        transformer;            % a data transformer object, (e.g. pcaTransformer)
    end
    
    methods
        
        function obj = GenerativeClassifierDist(varargin)
            [obj.transformer,obj.classConditionals,obj.classPrior] = process_options(varargin,...
                'transformer'           ,[],...
                'classConditionals'     ,[],...
                'classPrior'            ,[]);
        end
              
        
        function obj = fit(obj,varargin)
          % Fit the classifier
          %
          % FORMAT:
          %           obj = fit(obj,'name1',val1,'name2',val2,...)
          % INPUT:
          %           'X'            - X(i,:) is i'th case
          %           'y'            - the training labels
          [X,y] = process_options(varargin,'X',[],'y',[]);
          obj.classPrior = fit(obj.classPrior, 'data',  colvec(y));
          if(~isempty(obj.transformer))
            [X,obj.transformer] = train(obj.transformer,X);
          end
          for c=1:length(obj.classConditionals)
            obj.classConditionals{c} = fit(obj.classConditionals{c},'data',X(y==obj.classPrior.support(c),:));
          end
        end
        
        function pred = predict(obj,X)
          % pred(i) = p(y|X(i,:), params)
          % pred is a DiscreteDist
          if(~isempty(obj.transformer))
            X = test(obj.transformer,X);
          end
          logpy = log(predict(obj.classPrior));
          n= size(X,1);
          C = length(obj.classConditionals);
          L = zeros(n,C);
          for c=1:C
            L(:,c) = sum(logprob(obj.classConditionals{c},X),2) + logpy(c);
          end
          L = L - repmat(logsumexp(L,2),1,C); 
          pred = DiscreteDist('mu', exp(L)', 'support', obj.classPrior.support);
        end
        
        function d = ndimensions(obj)
            d = ndimensions(obj.classConditionals{1});
        end
        
    end
    
    methods(Static = true)
        
        function testClass()
            Ntrain = 100; Ntest = 100;
            Nclasses = 10;
            %[Xtrain,Xtest,ytrain,ytest] = setupMnist(binary, Ntrain, Ntest);
            d = 5; pi = (1/Nclasses)*ones(1,Nclasses); % uniform class labels
            XtrainC = rand(Ntrain,d); XtestC = rand(Ntest,d);
            Xtrain01 = XtrainC>0.5; Xtest01 = XtestC>0.5;
            ytrain = sampleDiscrete(pi,Ntrain,1);
            ytest = sampleDiscrete(pi,Ntest,1);
            classCond = cell(1,Nclasses);
            for binary=0:1
              for bayes=0:1
                if binary
                 Xtrain = Xtrain01; Xtest = Xtest01;
                 prior = BetaDist(1,1);
                 if bayes
                   for c=1:Nclasses, classCond{c} = Bernoulli_BetaDist('prior', prior); end
                 else
                   for c=1:Nclasses, classCond{c} = BernoulliDist('prior', prior); end
                 end
                else
                  Xtrain = XtrainC; Xtest = XtestC;
                  prior = NormInvGammaDist('mu', 0, 'k', 0.01, 'a', 0.01, 'b', 0.01);
                  if bayes
                    for c=1:Nclasses, classCond{c} = Gauss_NormInvGammaDist(prior); end
                  else
                   for c=1:Nclasses, classCond{c} = GaussDist('prior', prior); end
                 end
                end
                %classCond = copy(classCond,1,10) % requires that the
                %constructor need no arguments
                if bayes
                  alpha = 1*ones(1,Nclasses);
                  classPrior = Discrete_DirichletDist(DirichletDist(alpha), 1:Nclasses);
                else
                  classPrior = DiscreteDist('support',1:Nclasses);
                end
                model = GenerativeClassifierDist('classPrior', classPrior, 'classConditionals', classCond);
                model = fit(model,'X',Xtrain,'y',ytrain);
                pred  = predict(model,Xtest);
                errorRate = mean(mode(pred) ~= ytest)
              end
            end
        end
        
        function testClass2()
          % Multivariate Gaussian class cond densities
            Ntrain = 100; Ntest = 100;
            Nclasses = 10;
            %[Xtrain,Xtest,ytrain,ytest] = setupMnist(binary, Ntrain, Ntest);
            d = 5; pi = (1/Nclasses)*ones(1,Nclasses); % uniform class labels
            Xtrain = rand(Ntrain,d); Xtest = rand(Ntest,d);
            ytrain = sampleDiscrete(pi,Ntrain,1);
            ytest = sampleDiscrete(pi,Ntest,1);
            classCond = cell(1,Nclasses);
            prior = MvnInvWishartDist('mu', zeros(d,1), 'Sigma', eye(d), 'dof', d+1, 'k', 0.01);
            for bayes=0:1
              if bayes
                for c=1:Nclasses, classCond{c} = Mvn_MvnInvWishartDist(prior); end
              else
                for c=1:Nclasses, classCond{c} = MvnDist([],[],'prior', prior); end
              end
              if bayes
                alpha = 1*ones(1,Nclasses);
                classPrior = Discrete_DirichletDist(DirichletDist(alpha), 1:Nclasses);
              else
                classPrior = DiscreteDist('support',1:Nclasses);
              end
              model = GenerativeClassifierDist('classPrior', classPrior, 'classConditionals', classCond);
              model = fit(model,'X',Xtrain,'y',ytrain);
              pred  = predict(model,Xtest);
              errorRate = mean(mode(pred) ~= ytest)
            end
        end
       
    end % Static methods
    
  
end

