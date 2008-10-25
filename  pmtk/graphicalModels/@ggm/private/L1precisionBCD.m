function [precMat, covMat] = L1precisionBCD(X, varargin)
% written by Mark Schmidt

[rho, useQP, verbose, junk] = process_options(...
    varargin, 'regularizer', [], 'useQP', 0, 'verbose', 0);

[precMat] = helper(cov(X), rho, useQP, verbose);
covMat = inv(precMat);

%%%%%%% 

function [X] = helper(sigma_emp,lambda,useQP,verbose)

optTol = 0.00001;
S = sigma_emp;
p = size(S,1);
rho = lambda;
maxIter = 10;
A = [eye(p-1,p-1);-eye(p-1,p-1)];
f = zeros(p-1,1);

% Initial W
W = S + rho*eye(p,p);

% Check for qp mex file
if exist('qpas') == 3
    qpSolver = @qpas;
    qpArgs = {[],[],[],[]};
else
    qpSolver = @quadprog;
    options = optimset('LargeScale','off','Display','none');
    qpArgs = {[],[],[],[],[],options};
end

for iter = 1:maxIter

    % Check Primal-Dual gap
    X = W^-1; % W should be PD
    gap = trace(S*X) + rho*sum(sum(abs(X))) - p;
    if verbose, fprintf('Iter = %d, OptCond = %.5f\n',iter,gap); end
    if gap < optTol
        if verbose, fprintf('Solution Found\n'); end
        break;
    end

    for i = 1:p
      
      noti = setdiff(1:p, i);
        if verbose
            X = W^-1; % W should be PD
            gap = trace(S*X) + rho*sum(sum(abs(X))) - p;
            fprintf('Column = %d, OptCond = %.5f\n',i,gap);
            if gap < optTol
                fprintf('Solution Found\n');
                break;
            end
        end

        % Compute Needed Partitions of W and S
        s_12 = S(noti,i);
        
        if useQP
            % Solve as QP
            H = 2*W(noti,noti)^-1;
            b = rho*ones(2*(p-1),1) + [s_12;-s_12];
            w = qpSolver((H+H')/2,f,A,b,qpArgs{:});
        else
            % Solve with Shooting
            W_11 = W(noti,noti);
            Xsub = sqrtm(W_11);
            ysub = Xsub\s_12;
	    % Xsub' * Xsub = W_11
	    % Xsub' * ysub = Xsub' * Xsub^{-1} s_12 = s_12
	    % We use 2*rho because the code has a factor of 2
            w = W_11*LassoShooting(Xsub,ysub,2*rho,'verbose',0);
        end

        % Un-Permute
        W(noti,i) = w;
        W(i,noti) = w';
    end
    %drawnow
end
