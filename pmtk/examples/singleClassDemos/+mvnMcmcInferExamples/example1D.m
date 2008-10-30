seed = 1;
setSeed(seed);
m = mvnDist;
d = 5;
m = mkRndParams(m, d);
x = randn(1,d);
V = [1 2]; H = setdiff(1:d, V);
model{1} = m; model{1}.stateInfEng = mvnExactInfer;
name{1}= 'exact';
model{2} = m; model{2}.stateInfEng = mvnMcmcInfer('method', 'gibbs');
name{2} = 'gibbs';
%model{3} = m; model{3}.stateInfEng = mvnMcmcInfer('method', 'gibbs2');
%name{3} = 'gibbs2';
model{3} = m; model{3}.stateInfEng = mvnMcmcInfer('method', 'mh', 'SigmaProposal', 0.1*m.Sigma);
name{3} = 'mh';
Nmethods = 3;
for i=1:Nmethods
    model{i} = enterEvidence(model{i}, V, x(V));
    query{i} = marginal(model{i}, H);
    fprintf('method %s\n', name{i})
    mu = mean(query{i})
    C = cov(query{i})
end