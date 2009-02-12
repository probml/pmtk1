%% Gibbs Sampling for image denoising

setSeed(0);

% Generate Data
sigma = 2; % noise level
% input matrix consisting of letter A. The body of letter
% A is made of 1's while the background is made of -1's.
img = imread('lettera.bmp'); 
[M,N] = size(img);
img = double(img);
m = mean(img(:));
img2 = +1*(img>m) + -1*(img<m); % -1 or +1
y = img2 + sigma*randn(size(img2)); %y = noisy signal

% Create model
% prior = Ising
J = 1; % coupling strenght
% Observation model
CPD  = MvnMixDist('distributions',{MvnDist(-1,sigma^2), MvnDist(+1,sigma^2)});
model = IsingGridDist(J, CPD);

% Infernece
model.infEng = GibbsIsingGridInfEng('Nsamples', 1000);
model = condition(model, 'visible', y);
postmean = mean(marginal(model, 'hidden'));
