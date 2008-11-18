%% Sample From the Distribution
small = false;
setSeed(0);
if small
    as = [0.01 0.1 1]; bs = as;
else
    as = [0.1 0.5 1 2];
    bs = 1*ones(1,length(as));
end
figure;
for i=1:length(as)
    a = as(i); b = bs(i);
    XX = sample(InvGammaDist(a,b), 1000);
    subplot(length(as),1,i);
    hist(XX);
    title(sprintf('a=%4.3f,b=%4.3f', a, b))
end

