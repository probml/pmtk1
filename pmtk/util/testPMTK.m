%% Test PMTK

try
%% 
bernoulliDistTest;                     cls;
bernoulli_betaSequentialUpdate;        cls;
binomialPlotDemo;                      cls;
binomial_betaPosteriorDemo;            cls;
binom_betaPostPredDemo;                cls;
cancerRatesEb;                         cls;
chainTransformerTest;                  cls;
chordalGraphDemo;                      cls;
compareVarElimToEnum;                  cls;
compareVarElimToJtree;                 cls;
constDistTest;                         cls;
cooperYooInterventionDemo;             cls;
dataTableDemo;                         cls;
dirichletHistPlotDemo;                 cls;
discreteDistTest;                      cls;
gammaPlotDemo;                         cls;
gammaRainfallDemo;                     cls;
gaussDGMdemo;                          cls;
gaussInferMuSigmaDemo;                 cls;
gaussMixPlot;                          cls;
gauss_NormInvGammDistTest;             cls;
generativeClassifierTest1;             cls;
generativeClassifierTest2;             cls;
ggmBICdemo;                            cls;
ggmDemo;                               cls;
ggmInferDemo;                          cls;
gibbsSprinklerUGM;                     cls;
graphClassDemo;                        cls;
hmmDistTest;                           cls;
inheritedDiseaseVarElim;               cls;
invGammaSampleDemo;                    cls;
invWIplot1D;                           cls;
invWIplot2D;                           cls;
Knn3ClassHeatMaps;                     cls;
laplacePlotDemo;                       cls;
lingaussHybridDemo;                    cls;
linregAllMethods;                      cls;
linregGaussVsNIG;                      cls;
linreg_MvnDistTest;                    cls;
linreg_MvnInvGammaDistTest;            cls;
logregFitCrabs;                        cls;
logregSAT;                             cls;
logreg_MvnDistTest;                    cls;
mcmcMvn2dConditioning;                 cls;
misconceptionUGMdemo;                  cls;
mkAlarmNetworkDgm;                     cls;
mkFluDgm;                              cls;
mkMisconceptionUGM;                    cls;
mkSprinklerDgm;                        cls;
mvnImputationDemo;                     cls;
mvnPlot2Ddemo;                         cls;
%mvnSeqlUpdateMuSigma1D - broken!
mvnSeqUpdateSigma1d;                   cls;
mvnSoftCondition;                      cls;
mvtPlotDemo;                           cls;
numericalIntNIGdemo;                   cls;
oldFaithfulDemo;                       cls;
poissonPlotDemo;                       cls;
rainyDayDemo;                          cls;
sampleDistDemo;                        cls;
sampleHIWdemo;                         cls;
sprinklerDGMdemo;                      cls;
sprinklerUGMdemo;                      cls;
sprinklerUGMvarelim;                   cls;
studentVSGauss;                        cls;
undirectedChainFwdBackDemo;            cls;
WIplotDemo;                            cls;
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