function [x,f] = conicModel(funObj,x,options)

maxIter = 1000;
optTol = 1e-9;

p = length(x);

% Initialization of conic model
h = zeros(p,1);
J = eye(p);
H = eye(p);

% Evaluate Function
[f,g] = funObj(x);
funEvals = 1;

for i = 1:maxIter

    % Compute direction
    v = - H*J'*g;

    % Step size search
    if i == 1
       alpha = min(1,1/sum(abs(g))); 
    else
        alpha = 1;
    end
    x_alpha = x + (alpha*J*v)/(h'*alpha*v + 1);
    [f_alpha,g_alpha] = funObj(x_alpha);
    funEvals = funEvals + 1;
    s = x_alpha - x;
    row2 = (f - f_alpha)^2 - (g_alpha'*s)*(g'*s);
    
    if 0
        if f_alpha > f
            fprintf('Increasing...\n');
        end
    else
        % Step length is guaranteed to be a descent direction if H is
        % positive definite and J is non-singular
        while f_alpha > f
            fprintf('Backtracking...\n');
            alpha = alpha/2;
            x_alpha = x + (alpha*J*v)/(h'*alpha*v + 1);
            [f_alpha,g_alpha] = funObj(x_alpha);
            funEvals = funEvals + 1;
            s = x_alpha - x;
            row2 = (f - f_alpha)^2 - (g_alpha'*s)*(g'*s);
        end
    end

    while row2 < 0
        fprintf('Row2 < 0...\n');
        return
    end
    
    gamma = (-g'*s)/(f - f_alpha + sqrt(row2));
    v = v*alpha;

    % Update
    J = gamma*(J - s*h');
    h = ((1-gamma)/(gamma*g'*s))*J'*g;
    r = J'*g_alpha - (1/gamma)*(J +s*h')'*g;
    H = H + (v*(v-H*r)')/(v'*r) + ((v-H*r)*v')/(v'*r) - v*v'*(r'*(v-H*r))/((v'*r)^2);
    y = gamma*g_alpha - (1/gamma)*g;
    [R,p]=chol(H);
    if p ~= 0
        fprintf('Approximation not positive-definite\n');
        [v'*r s'*y 2*sqrt(row2)]
        return
    end

    % Take step
    x_old = x;
    f_old = f;
    x = x_alpha;
    f = f_alpha;
    g = g_alpha;

    fprintf('%10d %10d %15.5e %15.5e %15.5e\n',i,funEvals,alpha,f,sum(abs(g)));

    % Check convergence
    if sum(abs(x-x_old)) < optTol
        fprintf('Parameters changing by less than optTol\n');
        break;
    end
    if abs(f-f_old) < optTol
        fprintf('Objective changing by less than optTol\n');
        break;
    end
end