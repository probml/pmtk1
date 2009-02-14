function gibbsIsingImageDenoiseDemo()
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
CPDs = {MvnDist(-1,sigma^2), MvnDist(+1,sigma^2)};
model = IsingGridDist(J, CPDs);

% Inference
model.infEng = GibbsIsingGridInfEng('Nsamples', 50000, 'Nburnin', 1000, ...
  'progressFn', @plotter);
avgX = postMean(model, 'visible', y);

figure; imagesc(y); colormap('gray'); colorbar; title('noisy image');
figure; imagesc(avgX); colormap('gray'); colorbar; title('posterior mean');

 % plot intermediate results
  function plotter(X, iter)
    if rem(iter,10000) == 0,
      figure;
      imagesc(X);  axis('square'); colormap gray; axis off;
      title(sprintf('sample %d', iter));
      drawnow
    end
  end

end