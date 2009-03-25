%% Test PMTK

try


bernoulliDistTest;                       pclear(0);
bernoulli_betaSequentialUpdate;          pclear(0);
binom_betaPostPredDemo;                  pclear(0);
binomialPlotDemo;                        pclear(0);
binomial_betaPosteriorDemo;              pclear(0);
cancerRatesEb;                           pclear(0);
chainTransformerTest;                    pclear(0);
chordalGraphDemo;                        pclear(0);
compareVarElimToEnum;                    pclear(0);
compareVarElimToJtree;                   pclear(0);
constDistTest;                           pclear(0);
cooperYooInterventionDemo;               pclear(0);
dataTableDemo;                           pclear(0);
dirichletHistPlotDemo;                   pclear(0);
discreteDistTest;                        pclear(0);
gammaPlotDemo;                           pclear(0);
gammaRainfallDemo;                       pclear(0);
gaussDGMdemo;                            pclear(0);
gaussInferMuSigmaDemo;                   pclear(0);
gaussMixPlot;                            pclear(0);
gauss_NormInvGammDistTest;               pclear(0);
generativeClassifierTest1;               pclear(0);
generativeClassifierTest2;               pclear(0);
ggmBICdemo;                              pclear(0);
ggmDemo;                                 pclear(0);
ggmInferDemo;                            pclear(0);
gibbsSprinklerUGM;                       pclear(0);
graphClassDemo;                          pclear(0);
hmmDistTest;                             pclear(0);
inheritedDiseaseDemo;                    pclear(0);
invGammaSampleDemo;                      pclear(0);
invWIplot1D;                             pclear(0);
invWIplot2D;                             pclear(0);
jtreeSampleTest;                         pclear(0);
knn3ClassHeatMaps;                       pclear(0);
laplacePlotDemo;                         pclear(0);
lingaussHybridDemo;                      pclear(0);
linregAllMethods;                        pclear(0);
linregGaussVsNIG;                        pclear(0);
linreg_MvnDistTest;                      pclear(0);
linreg_MvnInvGammaDistTest;              pclear(0);
logregFitCrabs;                          pclear(0);
logregSAT;                               pclear(0);
logreg_MvnDistTest;                      pclear(0);
markovChainClassificationDemo;           pclear(0);
mcmcMvn2dConditioning;                   pclear(0);
misconceptionUGMdemo;                    pclear(0);
mkAlarmNetworkDgm;                       pclear(0);
mkFluDgm;                                pclear(0);
mkMisconceptionUGM;                      pclear(0);
mkSprinklerDgm;                          pclear(0);
mvnImputationDemo;                       pclear(0);
mvnPlot2Ddemo;                           pclear(0);
mvnSeqUpdateSigma1d;                     pclear(0);
mvnSoftCondition;                        pclear(0);
mvtPlotDemo;                             pclear(0);
numericalIntNIGdemo;                     pclear(0);
oldFaithfulDemo;                         pclear(0);
poissonPlotDemo;                         pclear(0);
rainyDayDemo;                            pclear(0);
sampleDistDemo;                          pclear(0);
sampleHIWdemo;                           pclear(0);
sprinklerDGMdemo;                        pclear(0);
sprinklerUGMdemo;                        pclear(0);
sprinklerUGMvarelim;                     pclear(0);
studentVSGauss;                          pclear(0);
undirectedChainFwdBackDemo;              pclear(0);
wiPlotDemo;                              pclear(0);

objectCreationTest; % try instantiating every class...
pclear(0);

catch ME
clc; close all
fprintf('PMTK Tests FAILED!\npress enter to see the error...\n\n');
pause
rethrow(ME)
end

cls
fprintf('PMTK Tests Passed\n')
