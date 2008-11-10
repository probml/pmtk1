%% Compare the Relative Performance of Various Optimizers
setSeed(1);
load soy; % n=307, d = 35, C = 3;
%load car; % n=1728, d = 6, C = 3;
methods = {'bb',  'cg', 'lbfgs', 'newton'};
lambda = 1e-3;
figure; hold on;

[styles, colors, symbols] =  plotColors;                                    %#ok

for mi=1:length(methods)
    tic
    [m, output{mi}] = fit(LogregDist, 'X', X, 'y', Y, ...
        'lambda', lambda, 'optMethod', methods{mi});                           %#ok
    T = toc                                                                 %#ok
    time(mi) = T;                                                           %#ok
    w{mi} = m.w;                                                            %#ok
    niter = length(output{mi}.ftrace)                                       %#ok
    h(mi) = plot(linspace(0, T, niter), output{mi}.ftrace, styles{mi});     %#ok
    legendstr{mi}  = sprintf('%s', methods{mi});                            %#ok
end
legend(legendstr)