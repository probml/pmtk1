function [x,f,exitflag,output] = minFuncTR(funObj,x0,options,varargin)
% solver can be: cauchy,steihaug,dogleg,schur,Linf
% cgSolve:
%   0: use analytic Hessian in steihaug
%   1: use Hv products

% Tunable parameters
[verbose,verboseI,maxIter,optTol,eta1,eta2,gamma1,gamma2,solver,Hessian,cgSolve,...
    HvFunc,useComplex,corrections] = myProcessOptions(options,...
    'verbose',1,'verboseI',1,'MaxIter',250,'optTol',1e-9,...
    'eta1',.25,'eta2',3/4,'gamma1',1/4,'gamma2',2,...
    'solver','schur','Hessian','exact','cgSolve',0,'HvFunc',[],'useComplex',0,'corrections',100);

if cgSolve ~= 0
    Hessian = 'off';
end

nVars = length(x0);
x = x0;

% Evaluate Initial Point
if strcmp(Hessian,'exact')
    [f,g,H] = funObj(x,varargin{:});
else
    [f,g] = funObj(x,varargin{:});
end
funEvals = 1;

if ~strcmp(Hessian,'exact')
    H = eye(nVars);
    if strcmp(Hessian,'lbfgs') || strcmp(Hessian,'lsr1')
        S = zeros(nVars,0);
        Y = zeros(nVars,0);
        Hdiag = 1;
    end
end

% TR parameters
delta = 100;
delta0 = 1e150;
lambda = 1;
normType = 2;

% Output Log
if verboseI
    fprintf('%10s %10s %15s %15s %15s %s\n','Iteration','FunEvals','TR-Radius','Function Val','Opt Cond','Step Type');
end

% Perform up to a maximum of 'maxIter' descent steps:
for i = 1:maxIter
    % Check Optimality Condition
    if sum(abs(g)) <= optTol
        exitflag=1;
        msg = 'Optimality Condition below optTol';
        break;
    end

    % Solve Trust-Region Sub-Problem

    if strcmp(solver,'cauchy') || (i == 1 && ~strcmp(Hessian,'exact'))
        % Use Cauchy Point
        if g'*H*g <= 0
            tau = 1;
        else
            tau = min(1,norm(g)^3/(delta*g'*H*g));
        end
        p = -tau*(delta/norm(g))*g;

    elseif strcmp(solver,'schur')
        % Solve Exactly with Binary Search for Ridge Regression Parameter
        % ***(could do search more effectively)***
        [Q,T] = schur(H);
        p = -(Q' \ ((Q \ g)./diag(T)));
        if norm(p) > delta
            fprintf('Regularizing\n');
            p = -(Q' \ ((Q \ g)./diag(T+lambda)));
            minLambda = [];
            maxLambda = [];
            while abs(norm(p) - delta)/delta > .1
                if norm(p) - delta > 0
                    minLambda = lambda;
                    if isempty(maxLambda)
                        lambda = lambda*2;
                    else
                        lambda = (lambda+maxLambda)/2;
                    end
                else
                    maxLambda = lambda;
                    if isempty(minLambda)
                        lambda = lambda/2;
                    else
                        lambda = (lambda+minLambda)/2;
                    end
                end
                p = -(Q' \ ((Q \ g)./diag(T+lambda)));
            end
        end

    elseif strcmp(solver,'dogleg')
        % Use piecewise-linear dog-leg path
        % ***(could do 2d subspace minimization)***
        % ***(does not always find intersection)***
        [R,posDef] = chol(H);

        if posDef == 0
            p = -R \ (R' \ g);
        else
            fprintf('Hessian not pd, modifying\n');
            H = H + eye(length(g)) * max(0,1e-12 - min(real(eig(H))));
            p = -H\g;
        end

        if norm(p) > delta
            pU = -g*(g'*g)/(g'*H*g);
            polyA = (p - pU)'*(p - pU);
            polyB = 2*(2*pU-p)'*(p - pU);
            polyC = (2*pU-p)'*(2*pU-p) - delta*delta;
            tau = max(roots([polyA polyB polyC]));
            if isempty(tau) || ~isreal(tau)
                fprintf('No Roots of Dogleg, using Cauchy\n');
                tau = min(1,norm(g)^3/(delta*g'*H*g));
                p = -tau*(delta/norm(g))*g;
            elseif tau <= 1
                fprintf('Using Steepest\n');
                p = tau*pU;
            else
                fprintf('Using DogLeg\n');
                p = pU + (tau-1)*(p - pU);
            end
        end
    elseif strcmp(solver,'steihaug')
        % ***(doesn't handle zero or negative curvature case)***
        % ***(could precondition the CG iterations)***
        if cgSolve == 0
            [p,j,res] = steihaugCG(H,g,delta,optTol,nVars,1);
            fprintf('SCG stopped after %d iterations with a residual of %.3f\n',j,res);
        else % finite-difference Hv products
            if isempty(HvFunc)
                useComplex
                HvArgs = {x,g,useComplex,funObj,varargin{:}};
                [p,j,res] = steihaugCG([],g,delta,optTol,nVars,1,[],[],@autoHv,HvArgs);
            else
                fprintf('User-supplied\n');
                HvArgs = {x,varargin{:}};
                [p,j,res] = steihaugCG([],g,delta,optTol,nVars,1,[],[],HvFunc,HvArgs);
            end
            fprintf('SCG stopped after %d iterations with a residual of %.3f\n',j,res);
            funEvals = funEvals+j;
        end
    elseif strcmp(solver,'Linf')
        % ***(could use a special-purpose bound-constrained solver)***
        if i == 1 || (i == 2 && ~strcmp(Hessian,'exact'))
            normType = inf;
            % Check for qpip
            if (exist('qpip','file')==3)
                useQPIP = 1;
            else
                useQPIP = 0;
                qpOps = optimset('Display','none','LargeScale','off','MaxIter',100000);
            end
        end
        bound = delta*ones(nVars,1);
        if useQPIP
            p = qpip(H,g,[],[],[],[],-bound,bound);
        else
            p = quadprog((H+H')/2,g,[],[],[],[],-bound,bound,[],qpOps);
        end
    elseif strcmp(solver,'L1')
        % ***(could use a special-purpose L1 solver)***
        if i == 1 || (i == 2 && ~strcmp(Hessian,'exact'))
            normType = 1;
            % Check for qpip
            if (exist('qpip','file')==3)
                useQPIP = 1;
            else
                useQPIP = 0;
                qpOps = optimset('Display','none','LargeScale','off','MaxIter',100000);
            end
        end
        LB = zeros(2*nVars,1);
        A = ones(1,2*nVars);
        b = delta;
        if useQPIP
            p = qpip([H -H;-H H],[g;-g],A,b,[],[],LB,[]);
        else
            p = quadprog([H -H;-H H],[g;-g],A,b,[],[],LB,[],[],qpOps);
        end
        p = p(1:nVars)-p(nVars+1:end);
    elseif strcmp(solver,'SPG')
        % Try to solve Trust-Region sub-problem exactly with SPG
        funObj_sub = @(p)subProblemObj(p,g,H);
        funProj = @(p)projectL2(p,delta);
        options_sub = options;
        options_sub.verbose = 0;
        p = minConF_SPG(funObj_sub,zeros(nVars,1),funProj,options_sub);
    else
        msg = 'Unknown Solver';
        exitflag = -3;
        break;
    end


    % Check Progress and Update Trust-Region
    f_old = f;

    if strcmp(Hessian,'off')
        if isempty(HvFunc)
            predVal = f + g'*p + (1/2)*p'*autoHv(p,HvArgs{:});
        else
            predVal = f + g'*p + (1/2)*p'*HvFunc(p,HvArgs{:});
        end
        funEvals = funEvals+1;
    else
        predVal = f + g'*p + (1/2)*p'*H*p;
    end

    if abs(f - predVal) <= optTol
        msg = 'Predicted change from solving trust-region sub-problem is below optTol';
        exitflag = -2;
        break;
    end

    if strcmp(Hessian,'exact')
        [f_new,g_new,H_new] = funObj(x+p,varargin{:});
    else
        [f_new,g_new] = funObj(x+p,varargin{:});
    end
    funEvals = funEvals + 1;

    row = (f - f_new)/(f - predVal);
    
    if row < eta1
        stepType2 = ',TR--';
        minDelta = 0;
        maxDelta = norm(p,normType)*gamma1;
        delta = maxDelta;
    else
        if row > eta2 && abs(norm(p,normType) - delta)/delta <= .1
            stepType2 = ',TR++';
            minDelta = delta;
            maxDelta = min(gamma2*delta,delta0);
            %delta = min(gamma2*delta,delta0);
            delta = maxDelta;
        else
            stepType2 = '';
            minDelta = gamma1*delta;
            maxDelta = delta;
            delta = maxDelta;
        end
    end

    if row >= eta1
        
        % Update Hessian
        if strcmp(Hessian,'exact');
            H = H_new;
        elseif strcmp(Hessian,'bfgs')
            y = g_new-g;
            s = p;
            ys = y'*s;
            if ys > 1e-10
                H = H + (y*y')/(ys) - (H*s*s'*H)/(s'*H*s);
            else
                fprintf('Skipping BFGS Update\n');
                pause;
            end
        elseif strcmp(Hessian,'sr1')
            y = g_new-g;
            s = p;
            ymHs = y-H*s;
            if sum(abs(s'*ymHs)) >= sum(abs(s))*sum(abs(ymHs))*1e-8
                H = H + (ymHs*ymHs')/(ymHs'*s);
            else
                fprintf('Skipping SR1 Update\n');
                pause;
            end
        elseif strcmp(Hessian,'lbfgs')
            y = g_new-g;
            s = p;
            ys = y'*s;

            if ys > 1e-10
                % Perform Update
                [S,Y,Hdiag] = lbfgsUpdate(y,s,corrections,1,S,Y,Hdiag);

                % Make Compact Representation
                k = size(Y,2);
                L = zeros(k);
                for j = 1:k
                    L(j+1:k,j) = S(:,j+1:k)'*Y(:,j);
                end
                N = [S/Hdiag Y];
                M = [S'*S/Hdiag L;L' -diag(diag(S'*Y))];
                H = (1/Hdiag)*eye(nVars) - N*(M\N');
            else
                fprintf('Skipping BFGS Update\n');
                pause;
            end
        elseif strcmp(Hessian,'lsr1')
            y = g_new-g;
            s = p;
            ys = y'*s;
            
            k = size(Y,2);
            L = zeros(k);
            for j = 1:k
                L(j+1:k,j) = S(:,j+1:k)'*Y(:,j);
            end
            D = diag(diag(S'*Y));
            N = Y-S/Hdiag;
            M = D+L+L'-S'*S/Hdiag;
            
            Bs= s/Hdiag - N*(M\(N'*s));
            ymBs = y-Bs;
            if sum(abs(s'*ymBs)) >= 1e-8*sum(abs(s))*sum(abs(ymBs))

                if k < corrections
                    % Full Update
                    S(:,k+1) = s;
                    Y(:,k+1) = y;
                else
                    % Limited-Memory Update
                    S = [S(:,2:corrections) s];
                    Y = [Y(:,2:corrections) y];
                end

                % Update scale of initial Hessian approximation
                Hdiag = (y'*s)/(y'*y);
                
                k = size(Y,2);
                L = zeros(k);
                for j = 1:k
                    L(j+1:k,j) = S(:,j+1:k)'*Y(:,j);
                end
                D = diag(diag(S'*Y));
                N = Y-S/Hdiag;
                M = D+L+L'-S'*S/Hdiag;
                
                H = (1/Hdiag)*eye(nVars) + N*(M\N');
            else
                fprintf('Skipping SR1 Update\n');
            end
        end
        
        x = x + p;
        f = f_new;
        g = g_new;
        stepType = 'Accept';

        % ******************* Check for lack of progress *******************

        if sum(abs(p)) <= optTol
            exitflag=2;
            msg = 'Step Size below optTol';
            break;
        end

        if abs(f-f_old) < optTol
            exitflag=2;
            msg = 'Function Value changing by less than optTol';
            break;
        end
    else
        stepType = 'Reject';
    end

    % Output iteration information
    if verboseI
        fprintf('%10d %10d %15.5e %15.5e %15.5e %s\n',i,funEvals,delta,f,sum(abs(g)),strcat(stepType,stepType2));
    end



    % ******** Check for going over iteration/evaluation limit
    % *******************

    if funEvals > maxIter
        exitflag = 0;
        msg='Exceeded Maximum Number of Iterations';
        break;
    end

end

% Output iteration information
if verboseI && exitflag ~= 0 && exitflag ~= 1 && exitflag ~= -2
    fprintf('%10d %10d %15.5e %15.5e %15.5e %s\n',i,funEvals,delta,f,sum(abs(g)),strcat(stepType,stepType2));
end

if verbose
    fprintf('%s\n',msg);
end
if nargout > 3
    output = struct('iterations',i,'funcCount',funEvals,...
        'firstorderopt',sum(abs(g)),'message',msg);
end

end


%% Trust-Region sub-problem objective function
function [f,g] = subProblemObj(w,g,H)

Hw = H*w;

f = (1/2)*w'*Hw + w'*g;
    
    if nargout > 1
       g = Hw + g;
    end
end

function [w] = projectL2(w,tau)
nw = norm(w);
    if nw > tau
        w = w*(tau/nw);
    end
end


%% Steihaug's Conjugate-Gradient Method
function [p,j,res] = steihaugCG(B,g,delta,optTol,maxIter,verbose,precFunc,precArgs,matrixVectFunc,matrixVectArgs)

p = zeros(size(g));
r = g;
res = norm(r);
d = -r;

if res < optTol
    j = 0;
    return;
end

for j = 1:maxIter
    if nargin >= 9
        Bd = matrixVectFunc(d,matrixVectArgs{:});
    else
        Bd = B*d;
    end
    dBd = d'*Bd;
    if dBd <= 0
        fprintf('Negative or Zero Curvature Detected\n');
        pause;
    end
    alpha = (r'*r)/(dBd);
    p_new = p + alpha*d;

    if norm(p_new) >= delta
        fprintf('Outside Trust-Region\n');
        tau = max(roots([d'*d 2*d'*p p'*p-delta^2]));
        p = p + tau*d;
        return;
    end
    p = p_new;
    r_new = r + alpha*Bd;
    res = norm(r_new);

    if res < optTol*norm(g)
        return;
    end
    beta = (r_new'*r_new)/(r'*r);
    r = r_new;
    d = -r_new + beta*d;
end
end