%% Simple Test of the Generative Classifier Class (2)
% Multivariate Gaussian class cond densities
%#testPMTK
Ntrain = 100; Ntest = 100;
Nclasses = 10;
d = 5; pi = (1/Nclasses)*ones(1,Nclasses); % uniform class labels
Xtrain = rand(Ntrain,d); Xtest = rand(Ntest,d);
ytrain = sampleDiscrete(pi,Ntrain,1);
ytest = sampleDiscrete(pi,Ntest,1);
classCond = cell(1,Nclasses);
prior = MvnInvWishartDist('mu', zeros(d,1), 'Sigma', eye(d), 'dof', d+1, 'k', 0.01);
priorC = DirichletDist(1*ones(1,Nclasses));
for bayes=0:1
    if bayes
        classCond = copy(Mvn_MvnInvWishartDist(prior),1,Nclasses);
    else
        classCond = copy(MvnDist('-ndims',d,'-prior', prior),1,Nclasses);
    end
    if bayes      
        classPrior = Discrete_DirichletDist(priorC);
    else
        classPrior = DiscreteDist('-nstates', Nclasses, '-prior', priorC);
    end
    model = GenerativeClassifierDist('-classPrior', classPrior, '-classConditionals', classCond);
    model = fit(model,'X',Xtrain,'y',ytrain);
    pred  = predict(model,Xtest);
end