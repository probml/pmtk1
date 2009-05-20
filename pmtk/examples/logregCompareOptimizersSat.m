setSeed(0);
stat = load('satData.txt');

y = stat(:,1);                      % class labels
X = stat(:,4);                      % SAT scores
[X,perm] = sort(X,'ascend');        % sort for plotting purposes
y = y(perm); % 0,1

n = size(X,1);
X = [ones(n,1) X];

winit = zeros(2,1);
lambda = 1e-3;
y01 = y;
addOffset = true;

%NLLfn = @LogregBinaryL2.logregNLLgradHess;
NLLfn = @logregNLLgradHess;
options = [];
options = optimset('GradObj','on','Hessian','on','Display','iter','DerivativeCheck','off'); 
[beta, err] = fminunc(NLLfn, winit, options, X, y01, lambda, addOffset)

fprintf('\n minfunc\n');
objective = @(w) NLLfn(w, X, y01, lambda, addOffset);
options.Method = 'lbfgs';
%options.Method = 'newton';
options.Display = true;
options.derivativeCheck = 'off' % 'on';
[w, f, exitflag, output] = minFunc(objective, winit, options)

[beta w]

