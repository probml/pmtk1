function [x,f,exitflag,output] = minFuncDFO(funObj,x0,options,varargin)
% minFuncDFO(funObj,x0,options,varargin)
%
% derivative-free optimization
%
% solver:
%   random - takes random steps, increases step lengths when they succeed,
%   decreases if they fail (used as sub-routine to initialize other
%   methods)
%
%   numDiff - calls minFunc with numerical differentiation on
%
%   interpModel - uses interpolation to build an approximation of the
%   gradient (requires quadratic storage)
%
%   coordinateSearch - cycles through the coordinates doing 1D line
%   minimizations (applicable to non-differentiable problems)
%
%   hookeJeeves - coordinateSearch, where the line connecting the first and
%   last points is searched at the end of each cycle (applicable to
%   non-differentiable problems)
%
%   patternSearch - cycles through and tries a set of patterns (applicable to
%   non-differentiable problems)
%
%   conjugateDirection - Powell's conjugate direction method (requires
%   quadratic storage)
%
%   nelderMead - Nelder and Mead downhill simplex method (requires
%   quadratic storage)
%
% lineMin:
%   0 : golden section search
%   1 : brent's method
%   -1: fminbnd


% Tunable parameters
[verbose,verboseI,maxIter,tolX,solver,useComplex,bracket,lineMin] = myProcessOptions(options,...
    'verbose',1,'verboseI',1,'MaxIter',250,'tolX',1e-9,...
    'solver','numDiff','useComplex',0,'bracket',2,'lineMin',4);

nVars = length(x0);
x = x0;
gold = 1.618034;

% Most algorithms don't have termination conditions implemented,
% so exceeding the maximum number of iterations is the defaults
msg='Exceeded Maximum Number of Iterations';
exitflag = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(solver,'random')

    [x,f,funEvals,msg,delta] = randomSolver(x,tolX,maxIter,funObj,varargin{:});

    i = funEvals-1;
    g = delta;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

elseif strcmp(solver,'numDiff')
    % Finite-Difference Approximation to Gradient

    % Turn on numerical differentation
    options.numDiff = 1;

    % Call solver
    [x,f,exitflag,output] = minFunc(funObj,x0,options,varargin{:});

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(solver,'interpModel')
    % Use a model that interpolates function values at different points

    % Start by Moving around randomly for nVars iterations
    [x,f,funEvals,msg,delta,F,Y] = randomSolver(x,0,min(nVars+1,maxIter),funObj,varargin{:});

    % Initialize Trust-Region Parameters
    delta = 1;
    delta0 = 1e150;
    eta = 1/4;
    perturbScale = 100;

    % Output Log
    if verboseI
        fprintf('%10s %10s %15s %15s %15s %s\n','Iteration','FunEvals','TR-Radius','Function Val','Opt Cond','Step Type');
    end

    % Start at minimum point in interpolation set
    [f,minPos] = min(F);
    x = Y(:,minPos);

    i = 1;
    while funEvals < maxIter

        % Form linear model to satisfy interpolation condition
        S = Y - repmat(x,[1 nVars+1]);
        Fdif = F - f;
        g = S'\Fdif;

        % Approximately solve Trust-Region sub-problem
        p = -g*delta/norm(g);

        % Compute Trial Point
        xTrial = x + p;
        fTrial = funObj(xTrial,varargin{:});
        funEvals = funEvals+1;

        % Check Actual vs. Predicted Reduction
        predVal = f + g'*p;
        row = (f - fTrial)/(f - predVal);

        if row >= eta
            % Accept point and expand trust-region radius
            stepType = 'Accept';
            stepType2 = ',TR++';

            delta = 1.5*delta;

            [junk,maxPos] = max(F);
            Y(:,maxPos) = xTrial;
            F(maxPos) = fTrial;

        elseif fTrial < f && cond(S') < 1e4
            % Accept point and reduce trust-region radius
            stepType = 'Accept';
            stepType2 = ',TR--';

            delta = delta*.75;

            [junk,maxPos] = max(F);
            Y(:,maxPos) = xTrial;
            F(maxPos) = fTrial;
        else
            % Update interpolation set
            stepType = 'Reject';
            stepType2 = '';
            [junk,maxPos] = max(F);
            Y(:,maxPos) = Y(:,minPos)+randn(nVars,1)*delta;
            F(maxPos) = funObj(Y(:,maxPos),varargin{:});
            funEvals = funEvals+1;

            delta = delta*.9;

        end

        % Compute new minimum point in interpolation set
        [f,minPos] = min(F);
        x = Y(:,minPos);

        % Output Log
        if verboseI
            fprintf('%10d %10d %15.5e %15.5e %15.5e %s\n',i,funEvals,delta,f,sum(abs(g)),strcat(stepType,stepType2));
        end

        if delta < tolX
            msg = 'Trust-Region Radius below tolX';
            break;
        end

        i = i + 1;
    end



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(solver,'coordinateSearch') || strcmp(solver,'hookeJeeves')

    if strcmp(solver,'hookeJeeves')
        HJ = 1;
    else
        HJ = 0;
    end

    if verboseI
        fprintf('%10s %10s %10s %15s\n','passes','iteration','funEvals','f');
    end
    f = funObj(x,varargin{:});
    funEvals = 1;

    i = 1;
    c = 0;

    x_old = x;
    f_old = f;
    scale = ones(nVars+1,1);
    while funEvals < maxIter

        if HJ
            c = c + 1;
            if c == nVars+2
                c = 1;
            elseif c == nVars+1
                % Hooke-Jeeves Step
                d = x-x_old;

                if sum(abs(x-x_old)) < tolX
                    msg = 'Parameter change after cycle less than tolX';
                    break;
                elseif sum(abs(f-f_old)) < tolX
                    msg = 'Function value change after cycle less than tolX';
                    break;
                end

                x_old = x;
                f_old = f;
            else % Otherwise Pick Coordinate
                d = zeros(nVars,1);
                d(c) = 1;
            end
        else
            % Pick Coordinate
            c = c + 1;
            if c == nVars+1
                c = 1;

                if sum(abs(x-x_old)) < tolX
                    msg = 'Parameter change after cycle less than tolX';
                    break;
                elseif sum(abs(f-f_old)) < tolX
                    msg = 'Function value change after cycle less than tolX';
                    break;
                end
                x_old = x;
                f_old = f;
            end

            % Search direction is simply 1 variable
            d = zeros(nVars,1);
            d(c) = 1;
        end

        if c <= nVars
            xtemp = x(c);
        else
            xtemp = x;
        end
        
        % Search for a minimum along this direction
        [x,f,LSfunEvals] = lineMinimize(funObj,x,f,d,bracket,lineMin,scale(c),varargin{:});
        funEvals = funEvals+LSfunEvals;
        
        if c <= nVars
            scale(c) = max(abs(xtemp-x(c)),tolX);
        end
        
        % Output Log
        if verboseI
            fprintf('%10d %10d %10d %15.5e\n',floor(i/nVars),i,funEvals,f);
        end
        i = i + 1;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(solver,'patternSearch')

    % Make patterns (the unit vectors and their negations)
    d = sparse(nVars,nVars*2);
    for i = 1:nVars
        d(i,i*2-1) = 1;
        d(i,i*2) = -1;
    end

    gamma = 1;

    f = funObj(x,varargin{:});
    funEvals = 1;
    i = 1;

    while funEvals < maxIter
        if gamma < tolX
            msg = 'Step length below tolX';
            break;
        end

        accept = 0;
        acceptLast = 0;
        for p = 1:size(d,2);
            if mod(p,2) == 0 && acceptLast
                % Don't bother evaluating this move,
                %   which reverses the move just accepted
                acceptLast = 0;
            else
                acceptLast = 0;
                f_new = funObj(x+gamma*d(:,p),varargin{:});
                funEvals = funEvals+1;
                if f_new - f < 0
                    accept = 1;
                    acceptLast = 1;
                    x = x + gamma*d(:,p);
                    f = f_new;
                    if mod(p,2) == 0
                        % Switch this pattern with the previous one
                        tmp = d(:,p-1);
                        d(:,p-1) = d(:,p);
                        d(:,p) = tmp;
                    end
                end
            end
        end

        if accept
            gamma = gamma*1.5;
            stepType = 'accept';
        else
            gamma = gamma*.25;
            stepType = 'reject';
        end

        if verboseI
            fprintf('%5d %5d %15.5e %15.5e %15s\n',i,funEvals,f,gamma,stepType);
        end
        i = i + 1;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(solver,'conjugateDirection')

    f = funObj(x,varargin{:});
    funEvals = 1;

    P = eye(nVars);

    if verboseI
        fprintf('%5s %5s %15s %15s\n','iter','funEvals','f','step size');
    end

    [x,f,LSfunEvals] = lineMinimize(funObj,x,f,P(:,nVars),bracket,lineMin,[],varargin{:});
    funEvals = funEvals+LSfunEvals;

    i = 1;
    while funEvals < maxIter

        z(:,1) = x;
        fz(1) = f;
        maxDec = -inf;
        for j = 1:nVars
            [z(:,j+1),fz(j+1),LSfunEvals] = lineMinimize(funObj,z(:,j),fz(j),P(:,j),bracket,lineMin,[],varargin{:});
            if fz(j)-fz(j+1) > maxDec
                maxDec = fz(j)-fz(j+1);
                maxDecInd = j;
            end
            funEvals = funEvals+LSfunEvals;
        end

        % Check if we will update directions and move along average
        % direction
        f_0 = fz(1);
        f_N = fz(nVars+1);
        f_E = funObj(2*z(:,nVars+1) - z(:,1),varargin{:});
        funEvals = funEvals + 1;

        x = z(:,nVars+1);
        f = fz(nVars+1);
        if f_E < f_0
            if 2*(f_0 - 2*f_N + f_E)*((f_0 - f_N) - maxDec)^2 < maxDec*(f_0 - f_E)^2
                P(:,maxDecInd) = P(:,nVars);
                P(:,nVars) = z(:,nVars+1) - z(:,1);
                [x,f,LSfunEvals] = lineMinimize(funObj,z(:,nVars+1),fz(nVars+1),P(:,nVars),bracket,lineMin,[],varargin{:});
                funEvals = funEvals+LSfunEvals;
            end
        end

        fprintf('%5d %5d %15.5e %15.5e\n',i,funEvals,f,sum(abs(x-z(:,1))));

        if abs(f-fz(1)) < tolX
            msg = 'Change in function value below tolX';
            break;
        elseif sum(abs(x-z(:,1))) < tolX
            msg = 'Step size below tolX';
            break;
        end

        i = i + 1;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(solver,'nelderMead')

    % Make initial simplex
    nVars = length(x);
    X = [x eye(nVars)];
    F(1) = funObj(x,varargin{:});

    scale = max(norm(x,inf),1);
    alpha = scale / (nVars*sqrt(2)) * [ sqrt(nVars+1)-1+nVars sqrt(nVars+1)-1 ];
    X(:,2:nVars+1) = (x + alpha(2)*ones(nVars,1)) * ones(1,nVars);
    for j=2:nVars+1
        X(j-1,j) = x(j-1) + alpha(1);
        x(:) = X(:,j);
        F(j) = funObj(x,varargin{:});
    end
    funEvals = nVars+1;

    if verboseI
        fprintf('%5s %5s %15s %15s\n','iter','funEvals','f','step size');
    end

    % Find minimum, 2nd largest, and maximum points
    [sorted,sortedInd] = sort(F);
    f = sorted(1);
    fmax2 = sorted(nVars);
    fmax = sorted(nVars+1);
    minInd = sortedInd(1);
    maxInd2 = sortedInd(nVars);
    maxInd = sortedInd(nVars+1);

    i = 1;
    while funEvals < maxIter
        if verboseI
            X_old = X;
        end

        % Compute Centroid of all points except worse
        x_bar = sum(X(:,setdiff(1:nVars+1,maxInd)),2)/nVars;

        % Reflect around worst point
        t = -1;
        x_r = x_bar + t*(X(:,maxInd) - x_bar);
        f_r = funObj(x_r,varargin{:});
        funEvals = funEvals+1;

        if f <= f_r && f_r < fmax2
            % Neither best nor worse
            stepType = 'Addition';
            X(:,maxInd) = x_r;
            F(maxInd) = f_r;
        elseif f_r < f
            % New Best
            t = -2;
            x_r2 = x_bar + t*(X(:,maxInd) - x_bar);
            f_r2 = funObj(x_r2,varargin{:});
            funEvals = funEvals+1;
            if f_r2 < f_r
                stepType = 'Expansion';
                X(:,maxInd) = x_r2;
                F(maxInd) = f_r2;
            else
                stepType = 'Reflexion';
                X(:,maxInd) = x_r;
                F(maxInd) = f_r;
            end
        else
            % Still Worse

            contractionSuccessful = 0;
            if f_r < fmax % Should be between max and 2nd-max
                t = -1/2;
                x_r2 = x_bar + t*(X(:,maxInd) - x_bar);
                f_r2 = funObj(x_r2,varargin{:});
                funEvals = funEvals+1;
                if f_r2 < f_r
                    stepType = 'Outside Contraction';
                    X(:,maxInd) = x_r2;
                    F(maxInd) = f_r2;
                    contractionSuccessful = 1;
                end
            else
                t = 1/2;
                x_r2 = x_bar + t*(X(:,maxInd) - x_bar);
                f_r2 = funObj(x_r2,varargin{:});
                funEvals = funEvals+1;
                if f_r2 < fmax
                    stepType = 'Inside Contraction';
                    X(:,maxInd) = x_r2;
                    F(maxInd) = f_r2;
                    contractionSuccessful = 1;
                end
            end

            if ~contractionSuccessful
                stepType = 'Shrink';
                for j = setdiff(1:nVars+1,minInd)
                    X(:,j) = (1/2)*(X(:,minInd)+X(:,j));
                    F(j) = funObj(X(:,j),varargin{:});
                    funEvals = funEvals+1;
                end
            end
        end

        [sorted,sortedInd] = sort(F);
        f = sorted(1);
        fmax2 = sorted(nVars);
        fmax = sorted(nVars+1);
        minInd = sortedInd(1);
        maxInd2 = sortedInd(nVars);
        maxInd = sortedInd(nVars+1);

        simplexSize = sum(sum(abs(X(:,sortedInd(2:end)) - repmat(X(:,sortedInd(1)),[1 nVars]))));

        if verboseI
            fprintf('%5d %5d %15.5e %15.5e %15.5e %s\n',i,funEvals,f,sum(abs(X(:)-X_old(:))),simplexSize,stepType);
        end

        if simplexSize <= tolX
            msg = 'Simplex size less than tolX';
            break;
        end

        i = i + 1;
    end

end

if ~strcmp(solver,'numDiff')
    if verbose
        fprintf('%s\n',msg);
    end
    if nargout > 3
        output = struct('iterations',i,'funcCount',funEvals,...
            'message',msg);
    end
end

end

%%
function [x,f,funEvals,msg,delta,traceF,traceX] = randomSolver(x,optTol,maxIter,funObj,varargin);

nVars = length(x);

fprintf('%10s %15s %15s %s\n','FunEvals','Step Scale','Function Val','Step Type');

delta = 1;
delta0 = 1e150;

f = funObj(x,varargin{:});
funEvals = 1;

if nargout >= 5
    traceF(1,1) = f;
end
if nargout >= 6
    traceX(:,1) = x;
end

prev = 0;

msg = 'Exceeded Maximum Number of Iterations';
while 1

    xTrial = x + randn(nVars,1)*delta;
    fTrial = funObj(xTrial,varargin{:});
    funEvals = funEvals + 1;

    if nargout >= 5
        traceF(funEvals,1) = fTrial;
    end
    if nargout >= 6
        traceX(:,funEvals) = xTrial;
    end

    stepType2 = '';
    if fTrial < f
        stepType = 'Accept';
        x = xTrial;
        f = fTrial;
        if prev == 1
            delta = min(2*delta,delta0);
            stepType2 = ',SS++';
        end
        prev = 1;
    else
        stepType = 'Reject';
        if prev == -1
            delta = delta/2;
            stepType2 = ',SS--';
        end
        prev = -1;
    end

    fprintf('%10d %15.5e %15.5e %s\n',funEvals,delta,f,strcat(stepType,stepType2));

    if funEvals >= maxIter
        break;
    end

    if delta < optTol^2
        msg = 'Step Size below tolX^2';
        break;
    end
end
end

%%
function f = wrapFunObj(t,funObj,x,d,varargin)
f = funObj(x+t*d,varargin{:});
end

function [x,f,funEvals] = lineMinimize(funObj,x0,f0,d,bracket,lineMin,scale,varargin)
% Finds a minimum of funObj along direction d

% Bracket Solution
if bracket == 0
    [a,b,c,fa,fb,fc,funEvals1] = bracketNR(@wrapFunObj,0,f0,funObj,x0,d,varargin{:});
else
    if ~isempty(scale)
        options.scale = scale;
    else
        options = [];
    end
    
    if bracket == 1
        [a,b,c,fa,fb,fc,funEvals1] = myBracket(@wrapFunObj,options,0,f0,funObj,x0,d,varargin{:});
    else
        options.bracket = bracket;
        [a,b,c,fa,fb,fc,funEvals1] = myBracket2(@wrapFunObj,options,0,f0,funObj,x0,d,varargin{:});
    end
end

if funEvals1 > 5
[a b c]
end

% Find Minimum in Bracket
if lineMin == 0
    [t,f,funEvals2]=golden(@wrapFunObj,a,b,c,fb,funObj,x0,d,varargin{:});
elseif lineMin == 1
    [t,f,funEvals2]=brent(@wrapFunObj,a,b,c,fb,funObj,x0,d,varargin{:});
elseif lineMin == 2
    [t,f,exitflag,output] = fminbnd(@wrapFunObj,a,c,[],funObj,x0,d,varargin{:});
    funEvals2 = output.funcCount;
elseif lineMin == 3
    [t,f,funEvals2]=mySearch(@wrapFunObj,a,b,c,fa,fb,fc,funObj,x0,d,varargin{:});
else
    [t,f,funEvals2]=mySearch2(@wrapFunObj,a,b,c,fa,fb,fc,funObj,x0,d,varargin{:});
end

% if funEvals1 > 5
% t
% pause;
% end

x = x0+t*d;
funEvals = funEvals1+funEvals2;
end


%%
function [a,b,c,fa,fb,fc,funEvals] = bracketNR(funObj,a,fa,varargin)
% Based on mnbrak function in numerical recipes

gold = 1.618034;
glimit = 100;
tiny = 1e-20;

b = randn;
fb = funObj(b,varargin{:});
funEvals = 1;
    
if fb > fa
    [a,b,fa,fb] = deal(b,a,fb,fa);
end

c = b + gold*(b-a);
fc = funObj(c,varargin{:});
funEvals = funEvals+1;

i = 1;
while fb > fc
    %fprintf('Iteration %d\n',i);
    i=i+1;

    r = (b-a)*(fb-fc);
    q = (b-c)*(fb-fa);
    u = b-((b-c)*q-(b-a)*r)/(2*sign(q-r)*max(abs(q-r),tiny));
    ulim = b+glimit*(c-b);

    funEvals = funEvals+1;
    if (b-u)*(u-c) > 0
        fu = funObj(u,varargin{:});
        if fu < fc
            % Minimum between b and c
            a = b;
            fa = fb;
            b = u;
            fb = fu;
            return;
        elseif fu > fb
            % Minimum between a and u
            c = u;
            fc = fu;
            return;
        end
        % Interpolation did not work
        u = c+gold*(c-b);
        fu = funObj(u,varargin{:});
        funEvals = funEvals+1;
    elseif (c-u)*(u-ulim) > 0
        % Interpolation between c and limit
        fu = funObj(u,varargin{:});
        if fu < fc
            b = c;
            c = u;
            u = c+gold*(c-b);
            fb = fc;
            fc = fu;
            fu = funObj(u,varargin{:});
            funEvals = funEvals+1;
        end
    elseif (u-ulim)*(ulim-c) >= 0
        % Interpolation went past limit
        u = ulim;
        fu = funObj(u,varargin{:});
    else
        % Bad Interpolation
        u = c + gold*(c-b);
        fu = funObj(u,varargin{:});
    end
    a = b;
    b = c;
    c = u;
    fa = fb;
    fb = fc;
    fc = fu;

    if a > c
        temp = [a fa];
        a = c;
        fa = fc;
        c = temp(1);
        fc = temp(2);
    end
end
end

%%
function [xmin,fmin,funEvals] = golden(funObj,a,b,c,fb,varargin)
% a,b,c in ascending order, with fb < fa and fb < fc

R = 0.61803399;
C = 1-R;
optTol = 1e-4;
maxIter = 100;

x0 = a;
x3 = c;

if abs(c-b) > abs(b-a)
    x1 = b;
    f1 = fb;
    x2 = b+C*(c-b);
    f2 = funObj(x2,varargin{:});
else
    x2 = b;
    f2 = fb;
    x1 = b-C*(b-a);
    f1 = funObj(x1,varargin{:});
end
funEvals = 1;

i = 1;
while i < maxIter && abs(x3-x0) > optTol*(abs(x1)+abs(x2))
    %fprintf('Iteration %d\n',i);
    i = i + 1;

    if f2 < f1
        x0 = x1;
        x1 = x2;
        f1 = f2;
        x2 = R*x1+C*x3;
        f2 = funObj(x2,varargin{:});
    else
        x3 = x2;
        x2 = x1;
        f2 = f1;
        x1 = R*x2+C*x0;
        f1 = funObj(x1,varargin{:});
    end
    funEvals = funEvals+1;
end

if f1 < f2
    xmin = x1;
    fmin = f1;
else
    xmin = x2;
    fmin = f2;
end
end

%%
function [xmin,fmin,funEvals] = brent(funObj,ax,bx,cx,fb,varargin)
% ax,bx,cx in ascending order, with fb < fa and fb < fc

optTol = 1e-4;
epsilon = 1e-10;
maxIter = 100;
gold = 0.381966;
funEvals = 0;

e = 0;
a = ax;
b = cx;

x = bx;
fx = fb;
w = bx;
fw = fb;
v = bx;
fv = fb;

i = 1;
while i < maxIter
    %fprintf('Iteration %d\n',i);
    i = i + 1;

    xm = .5*(a+b);
    tol1 = optTol*abs(x)+epsilon;
    tol2 = 2*tol1;
    if abs(x-xm) <= (tol2-.5*(b-a))
        xmin = x;
        fmin = fx;
        return;
    end
    if abs(e) > tol1
        r = (x-w)*(fx-fv);
        q = (x-v)*(fx-fw);
        p = (x-v)*q-(x-w)*r;
        q = 2*(q-r);
        if q > 0
            p = -p;
        end
        q = abs(q);
        etemp = e;
        e = d;
        if (abs(p) >= abs(.5*q*etemp)) || (p <= q*(a-x)) || (p >= q*(b-x))
            if x >= xm
                e = a-x;
            else
                e = b-x;
            end
            d = gold*e;
        else
            d = p/q;
            u = x+d;
            if (u-a < tol2) || (b-u < tol2)
                d = abs(tol1)*sign(xm-x);
            end
        end
    else
        if x >= xm
            e = a-x;
        else
            e = b-x;
        end
        d = gold*e;
    end
    if abs(d) >= tol1
        u = x+d;
    else
        u = x+sign(d)*abs(tol1);
    end
    fu = funObj(u,varargin{:});
    funEvals = funEvals+1;

    if fu <= fx
        if u >= x
            a = x;
        else
            b = x;
        end
        v = w;
        w = x;
        x = u;
        fv = fw;
        fw = fx;
        fx = fu;
    else
        if u < x
            a = u;
        else
            b = u;
        end
        if (fu <= fw) || (w == x)
            v = w;
            w = u;
            fv = fw;
            fw = fu;
        elseif (fu <= fv) || (v == x) || (v == w)
            v = u;
            fv = fu;
        end
    end
end
fprintf('Exceeded maxIter\n');
xmin = x;
fmin = fx;
end