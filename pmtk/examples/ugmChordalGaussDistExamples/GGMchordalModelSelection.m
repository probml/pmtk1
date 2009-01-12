%% Pick best possible GGM on 4 nodes using exhaustive seach

n = 10; d = 4; seed = 0; graphType = 'loop';
Phi = 0.1*eye(d); delta = 5; % hyper-params
setSeed(seed);
Gtrue = UndirectedGraph('type', graphType, 'nnodes', d);
truth = UgmGaussDist(Gtrue, [], []);
Atrue = Gtrue.adjMat;
Graphlayout('adjMatrix',Atrue,'undirected',true);
set(gcf,'name','True Graph');
truth = mkRndParams(truth);
Y = sample(truth, n);

prec{1} = inv(truth.Sigma); names{1} = 'truth';
prec{2} = inv(cov(Y)); names{2} = 'emp';
obj = UgmGaussChordalDist([], HiwDist([], delta, Phi), []);

scoreFunction = @(modelDist,model)logmarglik(model,'data',modelDist.Ydata);
modelSpace = mkAllGgmDecomposable(obj)';
scoreTransformer = @(score)exp(score);
md = fit(ModelDist('scoreFunction'   ,scoreFunction   ,...
                   'Ydata'           ,Y               ,...
                   'scoreTransformer',scoreTransformer,...
                   'models'          ,modelSpace      ,...
                   'ordering'        ,'descend'       , ...
                   'verbose', false)); % KPM

map = md.mapEstimate;
Graphlayout('adjMatrix',map.G.adjMat,'undirected',true);
set(gcf,'name','MAP');