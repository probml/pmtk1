%% Compare various Logreg Optimizers
%dataset = 'documents';
dataset = 'soy';

setSeed(1);
switch dataset
    case 'documents'
        load docdata; % n=900, d=600, C=2in training set
        y = ytrain;
        X = xtrain;
        methods = {'bb',  'cg', 'lbfgs', 'newton'};
    case 'soy'
        load soy; % n=307, d = 35, C = 3;
        y = Y; % turn into a binary classification problem by combining classes 1,2
        y(Y==1) = 2;
        y(Y==2) = 2;
        y(Y==3) = 1;
        methods = {'bb',  'cg', 'lbfgs', 'newton',  'boundoptRelaxed'};
end
lambda = 1e-3;
figure; hold on;
[styles, colors, symbols] =  plotColors;                                  
for mi=1:length(methods)
    tic
    [m, output{mi}] = fit(LogregDist, 'X', X, 'y', y, 'priorStrength', lambda, 'optMethod', methods{mi});   %#ok
    T = toc                                                                                        %#ok
    time(mi) = T;                                                                                   %#ok
    w{mi} = m.w;                                                                                    %#ok
    niter = length(output{mi}.ftrace)                                                                %#ok
    h(mi) = plot(linspace(0, T, niter), output{mi}.ftrace, styles{mi});                                %#ok
    legendstr{mi}  = sprintf('%s', methods{mi});                                                        %#ok
end
legend(legendstr)