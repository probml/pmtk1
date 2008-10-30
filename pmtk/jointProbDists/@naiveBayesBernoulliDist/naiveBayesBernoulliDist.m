classdef naiveBayesBernoulliDist < jointProbDist

    properties
        nclasses;                       % class labels must be in {1:n}
        classPrior;                     % dirichlet distribution
        featurePrior;                   % beta distribution
    end

    properties
        Nc;                             % Nc(c)    = sum(Y == c);
        Njc;                            % Njc(j,c) = sum(X(Y==c,:),1);
        classPosterior;                 % dirichlet distribution p(y|Nc,alpha)
        posteriorParamsMAP;
        posteriorParams;
    end



    methods

        function obj = naiveBayesBernoulliDist(varargin)
            obj.nclasses = process_options(varargin,'nclasses',[]);

        end


        function obj = fit(obj,varargin)
            [obj.classPrior,obj.featurePrior,X,Y] = process_options(varargin,'classPrior',[],'featurePrior',[],'X',[],'Y',[]);

            if(isempty(obj.nclasses))
                obj.nclasses = max(Y);
            end

            if(~all(ismember(unique(Y),1:obj.nclasses)))
                error('Y must be in {1:%d}',obj.nclasses);
            end

            if(isempty(obj.classPrior))
                obj.classPrior = dirichletDist(ones(1,obj.nclasses)); %uninformative prior
            end

            if(isempty(obj.featurePrior))
                obj.featurePrior = betaDist(1,1);                       %uninformative prior
            end

            obj.Nc = histc(Y,1:obj.nclasses);
            obj.Nc = reshape(obj.Nc,1,numel(obj.Nc));
            for c=1:obj.nclasses
                obj.Njc(:,c) = sum(X(Y==c,:),1);
            end

            obj.classPosterior = dirichletDist(obj.Nc + obj.classPrior.alpha);
            
            obj.posteriorParamsMAP = zeros(obj.nclasses,1);
            for c=1:obj.nclasses
                obj.posteriorParamsMAP(c) = (obj.featurePrior.a + obj.Njc(:,c) - 1)/(obj.Nc(c)+ obj.featurePrior.a + obj.featurePrior.b -2);
                obj.posteriorParams(c) = (obj.featurePrior.a + obj.Njc(:,c))/(obj.Nc(c)+ obj.featurePrior.a + obj.featurePrior.b);
            end


        end


        function [pred,obj] = predict(obj,varargin)
            % classMethod -   'map' | 'exact'
            % featureMethod - 'map' | 'exact'
            [X,classMethod,featureMethod] = process_options(varargin,'X',[],'classMethod','MAP','featureMethod','MAP');


            switch(lower(classMethod))

                case 'map'
                    classPostProbs = obj.classPosterior.mode();

                case 'exact'
                    classPostProbs = obj.classPosterior.mean();
            end


            logprobs = zeros(size(X,1),obj.nclasses);
            switch(lower(featureMethod))

                case 'map'  % plugin posterior mode - map approximation
                    for c=1:obj.nclasses
                        thetaMAP_c = (obj.featurePrior.a + obj.Njc(:,c) - 1)/(obj.Nc(c)+ obj.featurePrior.a + obj.featurePrior.b -2);
                        logprobs(:,c) = (X*log(thetaMAP_c) + (1-X)*log(1-thetaMAP_c)) + log(classPostProbs(c));
                    end
                case 'exact'  %plugin posterior mean (equivilent to full posterior predictive
                    for c=1:obj.nclasses
                        thetaMAP_c = (obj.featurePrior.a + obj.Njc(:,c))/(obj.Nc(c)+ obj.featurePrior.a + obj.featurePrior.b);
                        logprobs(:,c) = (X*log(thetaMAP_c) + (1-X)*log(1-thetaMAP_c)) + log(classPostProbs(c));
                    end

            end

            pred = discreteDist(exp(logprobs));

        end
    end


    methods(Static = true)

        function classTest()
            load votes;
            nb = naiveBayesBernoulliDist('nclasses',2);
            nb = fit(nb,'X',X,'Y',Y,'classPrior',dirichletDist([20,3]),'featurePrior',betaDist(15,10));
            [nb,pred] = predict(nb,'X',X,'classMethod','exact','featureMethod','exact');
            yhat = mode(pred);
            trainSetErrorRate = mean(yhat'~=Y)
        end



    end









end