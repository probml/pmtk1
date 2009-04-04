%% Cluster MNIST Digits and Visualize the Cluster Centers

% Setup data
setSeed(1);
classes   = 2:4;    % Confine task to digits 2:4 (must be a subset of 0:9)
%nexamples = 2000;   % max 60000
nexamples = 2000;
binary = true;
ntest = 0;
[X,j,y,j] = setupMnist(binary,nexamples,ntest);    %#ok
clear j;
ndx       = ismember(y,classes);
X         = double(X(ndx,:)); % 603 x 784, each entry is 0 or 1
clear y; % unsupervised!

% Fit
%m = fit(DiscreteMixDist('nmixtures',numel(lookfor)),'data',X,'nrestarts',1);
fitEng = EmEng('nrestarts', 1, 'verbose', true);
model = MixDiscrete('nmixtures', numel(classes), 'support', [0 1], 'fitEng', fitEng);
model = fit(model, X);

% Visualize fitted model
for i=1:numel(classes)
    figure;
    mu = pmf(model.distributions{i});
    imagesc(reshape(mu(2,:),28,28));
end
placeFigures('Square',true)

% Display Samples from model
if(0) 
    for i=1:numel(classes)
        figure;
        imagesc(reshape(mean(sample(model.distributions{i},500),1),28,28))
    end
    placeFigures;
end



