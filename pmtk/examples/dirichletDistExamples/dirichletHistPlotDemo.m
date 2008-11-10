%% Plot Histogram
alpha = 0.1; 
seed = 1; 
rand('twister', seed); randn('state', seed);
obj = DirichletDist(alpha*ones(1,5));
n = 5;
probs = sample(obj, n);
figure;
for i=1:n
    subplot(n,1,i); bar(probs(i,:))
    if i==1, title(sprintf('Samples from Dir %3.1f', alpha)); end
end