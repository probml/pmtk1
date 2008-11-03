%% Conditioning an MVN
setSeed(0);
d = 4;
obj = mkRndParams(mvnDist, d);
x = randn(d,1);
V = [3 4];
obj = enterEvidence(obj, V, x(V));
x(:)'
fprintf('j \t %8s \t %8s\n', 'mean', 'Var');
for j=1:d
    m = marginal(obj, j);
    fprintf('%d \t %8.3f \t %8.3f\n', j, mean(m), var(m));
end