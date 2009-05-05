%% Compare speed of different optimizers for logistic regressiion
%#broken
function logregCompareOptimizers()

setSeed(1);

load soy; % n=307, d = 35, C = 3;
methods = {'boundoptRelaxed', 'bb',  'cg', 'lbfgs', 'newton'};
helper(X,Y, methods);


%load car; % n=1728, d = 6, C = 3;

load docdata; % n=900, d=600, C=2
methods = {'boundoptRelaxed', 'bb',  'cg', 'lbfgs'};
helper(xtrain, ytrain, methods);
end

function helper(X,Y,methods)
lambda = 1e-3;
figure; hold on;
[styles] =  plotColors;          
[n,d] = size(X);
C = length(unique(Y));
nmethods = length(methods);
W = zeros(d, C-1, nmethods);
D = DataTable(X,Y);
for mi=1:length(methods)
    tic
    m = LogregL2('-lambda', lambda, '-optMethod', methods{mi});
    [m, output{mi}] = fit(m, D); %#ok
    T = toc    %#ok                                                           
    time(mi) = T;   %#ok     
    W(:,:,mi) = m.w;                                     
    niter = length(output{mi}.ftrace)     %#ok 
    %ndx = 5:niter;
    h(mi) = plot(linspace(0, T, niter), output{mi}.ftrace, styles{mi});   %#ok
    legendstr{mi}  = sprintf('%s', methods{mi});           %#ok                
end
legend(legendstr)
title(sprintf('n=%d, d=%d, C=%d', n, d, C));
 set(gca,'ylim',[ylim(1) 0.10*ylim(2)])
squeeze(W(:,1,:))
end
