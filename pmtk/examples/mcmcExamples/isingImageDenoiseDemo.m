function isingImageDenoiseDemo()
%% image denoising using Ising prior and Gibbs sampling or mean field

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
J = 1; % coupling strenght
CPDs = {MvnDist(-1,sigma^2), MvnDist(+1,sigma^2)};
model = IsingGridDist(J, CPDs);

folder = 'C:\kmurphy\PML\pdfFigures';
doPrint = false;

figure; imagesc(y);  colorbar; title('noisy image');
axis('square'); colormap gray; axis off; 
if doPrint
  fname = fullfile(folder, sprintf('isingImageDenoise.pdf'));
  pdfcrop; print(gcf, '-dpdf', fname);
end
   
methods = {'gibbs', 'meanfield', 'meanfieldPU'};

for m=1:length(methods)
  method = methods{m};
  args = {'maxIter', 25, 'progressFn', @plotter};
  if strcmp(method, 'meanfieldPU')
    methodName = 'meanfield'; 
    args(end+1:end+2) = {'parallelUpdates', 'true'}; 
  else
    methodName = method;
  end
  mu = postMean(model, y, 'infMethod', methodName, 'infArgs',args);
  
  figure; imagesc(mu); colormap('gray');
  colorbar; title(sprintf('posterior mean (%s)', method));
  axis('square'); colormap gray; axis off;
   if doPrint
     fname = fullfile(folder, sprintf('isingImageDenoise%sMean.pdf', method));
     pdfcrop; print(gcf, '-dpdf', fname);
   end
end

 % plot intermediate results
  function plotter(X, iter)
    if any(iter == [ 1, 3, 5, 10 ])
      figure;
      imagesc(X);  axis('square'); colormap gray; axis off; colorbar;
      title(sprintf('sample %d, %s', iter, method));
      drawnow
      if doPrint
        fname = fullfile(folder, sprintf('isingImageDenoise%s%d.pdf', method, iter));
        pdfcrop; print(gcf, '-dpdf', fname);
      end
    end
  end


end