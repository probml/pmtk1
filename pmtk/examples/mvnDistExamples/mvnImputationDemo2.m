%% Imputation for an MVN 

d = 10; seed = 0; pcMissing = 0.3;
setSeed(seed);
model = mkRndParams(MvnDist, d);
%model = condition(model);
n = 50;

% If we are always missing the first K columns, we can never estimate their
% params, even using EM... So we use a random missing pattern, which is
% harder to visualize.
Xfull = sample(model, n);
missing = rand(n,d) < pcMissing; % set a random subset to missing
%missing = zeros(n,d);
%missing(:, 1:floor(pcMissing*d)) = 1; % set first pc% columns to missing

Xmiss = Xfull;
Xmiss((missing)) = NaN;
XmissImg = Xmiss;
XmissImg((missing)) = 0; % cannot plotNaN's

figure; imagesc(Xfull); title('full data'); colorbar
figure; imagesc(XmissImg); title(sprintf('%3.2f pc missing', pcMissing)); colorbar
  
% We can either impute using the generating model
% or, more realistically, we can fit the model to the data 
% and then use the fitted model to impute

trueModel = model;
baseModel = mkRndParams(trueModel); % to prevent cheating

models = {trueModel, ...
  fit(baseModel, 'data', Xfull, 'prior', 'niw'), ...
  fit(baseModel, 'data', Xfull, 'prior', 'none'), ...
  fit(baseModel, 'data', Xmiss, 'prior', 'niw', 'fitArgs', {'verbose', true})
  };
 % fit(baseModel, 'data', Xmiss, 'prior', 'none')};
methods = {'true', 'obs MLE', 'obs MAP', 'EM MAP'};
for i=1:length(models)
  Ximpute = impute(models{i}, Xmiss); % all the work happens here
  figure;  imagesc(Ximpute); colorbar
  mse(i) = mean((Ximpute - Xfull).^2);
  title(sprintf('imputed with %s, mse %5.3f ', methods{i}, mse(i));
end
