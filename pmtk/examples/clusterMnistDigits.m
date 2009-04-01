%% Cluster MNIST Digits and Visualize the Cluster Centers
%% Setup Data
setSeed(1);
lookfor   = 2:4;    % Confine task to digits 2:4 (must be a subset of 0:9)
nexamples = 2000;   % max 60000
binary = true;
ntest = 0;
[X,j,y,j] = setupMnist(binary,nexamples,ntest);    %#ok
clear j;
ndx       = ismember(y,lookfor);
X         = double(X(ndx,:)); % 603 x 784, each entry is 0 or 1
clear y; % unsupervised!
%% Fit
m = fit(DiscreteMixDist('nmixtures',numel(lookfor)),'data',X,'nrestarts',1);
%% Visualize
for i=1:numel(lookfor)
    figure;
    mu = pmf(m.distributions{i});
    imagesc(reshape(mu(2,:),28,28));
end
placeFigures('Square',true)
%% Display Samples
if(0) 
    for i=1:numel(lookfor)
        figure;
        imagesc(reshape(mean(sample(m.distributions{i},500),1),28,28))
    end
    placeFigures;
end



