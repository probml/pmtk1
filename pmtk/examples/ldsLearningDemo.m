%% Parameter Learning in a Linear Dynamical System 
% In this example, we sample from a simple LDS and try to learn back the
% dynamics. 
%#testPMTK
%% Create 'Ground Truth'
stateSize = 4;  % Hidden states are 4D, perhaps 2d position and velocity
obsSize   = 2;  % 2D observations
sysMatrix = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1];
obsMatrix = [1 0 0 0; 0 1 0 0];
sysNoise  = MvnDist(zeros(stateSize,1),0.1*eye(stateSize));
obsNoise  = MvnDist(zeros(stateSize,1),eye(obsSize));
startDist = MvnDist([10;10;1;0],10*eye(stateSize));

groundTruth = LinearDynamicalSystem(...
    'sysMatrix' ,sysMatrix  ,...
    'obsMatrix' ,obsMatrix  ,...
    'sysNoise'  ,sysNoise   ,...
    'obsNoise'  ,obsNoise   ,...
    'startDist' ,startDist  );

%% Sample from 'Ground Truth'
setSeed(0);
nTimeSteps = 100;
[Z,Y] = sample(groundTruth,nTimeSteps);
%% Learn Back the Dynamics
testModel = LinearDynamicalSystem('stateSize',stateSize); % at a minimum, we must specify the dimensionality of the hidden states.
testModel = fit(testModel,Y,'verbose',true);

% Initializing the params to sensible values is crucial. Here we simply use
% random values, (bad idea!) Lack of identifiability means the learned
% params are often far from the true ones. All that EM guarantees is that 
% the likelihood will increase. The current values of the model, if any,
% are used to initialize EM. You can specify these, prior to fitting, just
% like we did for the 'Ground Truth' model above. 



