%% Chain Classification Example
% In this example, we use a generative classifier with Markov chains as the
% class conditional densities to classify non-iid sequences. 
%#testPMTK
%% Setup
setSeed(0);
nstates     = 10;       % number of discrete states
nclasses    = 5;        % number of classes
chainLength = 50;       % the length of each chain in the data
ntrain      = 300;      % the number of training sequences
ntest       = 20;       % the number of test sequences
%% Create Class Conditionals
classConditionals = cell(1,nclasses);
for c=1:nclasses
   classConditionals{c} = mkRndParams(MarkovDist(),nstates);   
end
%% Synthesize Data
Xtrain = zeros(ntrain*nclasses,chainLength);
ytrain = zeros(ntrain*nclasses,1);
Xtest  = zeros(ntest*nclasses,chainLength);
ytest  = zeros(ntest*nclasses,1);
for c=1:nclasses
   trainNDX = (c-1)*ntrain+1:(c-1)*ntrain + ntrain;
   testNDX  = (c-1)*ntest +1:(c-1)*ntest  + ntest;
   Xtrain(trainNDX,:) = sample(classConditionals{c},chainLength,ntrain);
   ytrain(trainNDX,1) = c;
   Xtest (testNDX, :) = sample(classConditionals{c},chainLength,ntest);
   ytest (testNDX, 1) = c;
end
%% Create Class Prior
classPrior = DiscreteDist(normalize(ones(nstates,1)));
%% Create Generative Classifier
classifier = GenerativeClassifierDist('classConditionals',classConditionals,'classPrior',classPrior);
%% Fit Classifier
classifier = fit(classifier,'X',Xtrain,'y',ytrain);
%% Classify Test Examples
pred       = predict(classifier,Xtest);
yhat       = mode(pred);
%% Assess Error Rate
err        = mean(yhat ~= ytest)        %#ok
