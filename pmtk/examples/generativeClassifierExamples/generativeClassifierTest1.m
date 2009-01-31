%% Simple Test of the Generative Classifier Class (1)
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