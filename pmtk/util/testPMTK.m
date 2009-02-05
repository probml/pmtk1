%% Test PMTK

try
%% bernoulliDist
bernoulliDistTest
close('all'); clear('all');
%% bernoulli_BetaDist
bernoulli_betaSequentialUpdate
close('all'); clear('all');

%% betaBinomDist
ebCancerExample
close('all'); clear('all');

%% binomDist
binomialPlotDemo
close('all'); clear('all');

%% binom_betaDist
binomial_betaPosteriorDemo
binom_betaPostPredDemo
close('all'); clear('all');

%% chainTransformer
chainTransformerTest
close('all'); clear('all');

%% chordalGraph
chordalGraphDemo
close('all'); clear('all');

%% constDist
constDistTest
close('all'); clear('all');

%% dataTable
demoDataTable
close('all'); clear('all');

%% dgmDist
cooperYooInterventionDemo
gaussDGMdemo
inheritedDiseaseVarElim
mkAlarmNetworkDgm
mkSprinklerDgm
rainyDayDemo
sprinklerDGMdemo
close('all'); clear('all');

%% dirichletDist
dirichletHistPlotDemo
close('all'); clear('all');

%% discreteDist
discreteDistTest
close('all'); clear('all');

%% enumInfEng
enumSprinkler
close('all'); clear('all');

%% gammaDist
gammaPlotDemo
gammaRainfallDemo
close('all'); clear('all');

%% gauss_NormInvGammaDist
gaussInferMuSigmaDemo
gauss_NormInvGammDistTest
close('all'); clear('all');

%% generativeClassifier
generativeClassifierTest1
generativeClassifierTest2
close('all'); clear('all');

%% graph
graphClassDemo
close('all'); clear('all');

%% hiwDist
sampleHIWdemo
close('all'); clear('all');

%% hmmDist
hmmDistTest
close('all'); clear('all');

%% invGammaDist
invGammaSampleDemo
close('all'); clear('all');

%% invWishartDist
invWIplot1D
invWIplot2D
close('all'); clear('all');

%% knnDist
Knn3ClassHeatMaps
close('all'); clear('all');

%% laplaceDist
laplacePlotDemo
close('all'); clear('all');

%% linregDist
linregAllMethods
linregGaussVsNIG
close('all'); clear('all');

%% linreg_MvnDist
linreg_MvnDistTest
close('all'); clear('all');

%% linreg_MvnInvGammaDist
linreg_MvnInvGammaDistTest
close('all'); clear('all');

%% logregDist
logregFitCrabs
logregSAT
close('all'); clear('all');

%% logreg_MvnDist
logreg_MvnDistTest
close('all'); clear('all');

%% mcmc
gibbsSprinklerUGM
mcmcMvn2dConditioning
close('all'); clear('all');

%% mvnDist
mvnImputationDemo
mvnPlot2Ddemo
mvnSoftCondition
close('all'); clear('all');

%% mvnMixDist
gaussMixPlot
oldFaithfulDemo
close('all'); clear('all');

%% mvn_InvWishartDist
mvnSeqUpdateSigma1d
close('all'); clear('all');

%% mvn_MvnInvWishartDist
mvnSeqlUpdateMuSigma1D
close('all'); clear('all');

%% mvtDist
mvtPlotDemo
close('all'); clear('all');

%% normalInvGammaDist
demoNumericalIntNIG
close('all'); clear('all');

%% poissonDist
poissonPlotDemo
close('all'); clear('all');

%% sampleDist
sampleDistDemo
close('all'); clear('all');

%% studentDist
studentVSGauss
close('all'); clear('all');

%% ugmGaussDist
ggmBICdemo
ggmDemo
ggmInferDemo
close('all'); clear('all');

%% ugmTabularDist
misconceptionUGMdemo
mkMisconceptionUGM
sprinklerUGMdemo
close('all'); clear('all');

%% varElimInfEng
compareToEnum
close('all'); clear('all');

%% wishartDist
WIplotDemo
close('all'); clear('all');

% try instantiating every class...
objectCreationTest

catch ME
	clc; close all
	fprintf('PMTK Tests FAILED!\npress enter to see the error...\n\n')
	pause
	rethrow(ME)
end

cls
fprintf('PMTK Tests Passed\n')