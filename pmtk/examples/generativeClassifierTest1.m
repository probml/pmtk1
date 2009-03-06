%% Simple Test of the Generative Classifier Class (1)
%#testPMTK
Ntrain = 100; Ntest = 100;
Nclasses = 10;
d = 5; pi = (1/Nclasses)*ones(1,Nclasses); % uniform class labels
XtrainC = rand(Ntrain,d); XtestC = rand(Ntest,d);
Xtrain01 = XtrainC>0.5; Xtest01 = XtestC>0.5;
ytrain = sampleDiscrete(pi,Ntrain,1);
ytest = sampleDiscrete(pi,Ntest,1);
for binary=0:1
    for bayes=0:1
        if binary
            Xtrain = Xtrain01; Xtest = Xtest01;
            prior = BetaDist(1,1);
            if bayes
                classCond = copy(Bernoulli_BetaDist('prior', prior),1,Nclasses);
            else
                classCond = copy(BernoulliDist('prior', prior),1,Nclasses);
            end
        else
            Xtrain = XtrainC; Xtest = XtestC;
            prior = NormInvGammaDist('mu', 0, 'k', 0.01, 'a', 0.01, 'b', 0.01);
            if bayes
                classCond = copy(Gauss_NormInvGammaDist(prior),1,Nclasses);
            else
                classCond = copy(GaussDist('prior', prior),1,Nclasses);
            end
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
    end
end