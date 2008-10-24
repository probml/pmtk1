function [d,old_dirs,old_stps,Hdiag] = sr1Update(g,y,s,corrections,debug,old_dirs,old_stps,Hdiag)

%B0 = eye(length(y))/Hdiag;
S = old_dirs;
Y = old_stps;
k = size(Y,2);
L = zeros(k);
for j = 1:k
    for i = j+1:k
        L(i,j) = S(:,i)'*Y(:,j);
    end
end
D = diag(diag(S'*Y));
N = Y-S/Hdiag;
M = D+L+L'-S'*S/Hdiag;

Bs = s/Hdiag - N*(M\(N'*s)); % Product B*s

ymBs = y-Bs;
if sum(abs(s'*ymBs)) >= 1e-8*sum(abs(s))*sum(abs(ymBs))
    fprintf('Performing SR1 Update\n');

    numCorrections = size(old_dirs,2);
    if numCorrections < corrections
        % Full Update
        old_dirs(:,numCorrections+1) = s;
        old_stps(:,numCorrections+1) = y;
    else
        % Limited-Memory Update
        old_dirs = [old_dirs(:,2:corrections) s];
        old_stps = [old_stps(:,2:corrections) y];
    end

    % Update scale of initial Hessian approximation
    Hdiag = (y'*s)/(y'*y);
else
    fprintf('Skipping SR Update\n');
    pause;
end

S = old_dirs;
Y = old_stps;
k = size(Y,2);
L = zeros(k);
for j = 1:k
    for i = j+1:k
        L(i,j) = S(:,i)'*Y(:,j);
    end
end
D = diag(diag(S'*Y));
N = Y-S/Hdiag;
M = D+L+L'-S'*S/Hdiag;

% Now Compute Descent Direction
[n,m] = size(S);

B = eye(n)/Hdiag + N*M^-1*N';
[R,posdef] = chol(B);

%[R,posdef] = chol(-(M + N'*Hdiag*N));
if posdef == 0
    fprintf('SR1 Update Sufficiently PD\n');
    d = -Hdiag*g + Hdiag*N*((M + N'*Hdiag*N)\(N'*Hdiag*g));
else
    fprintf('SR1 Update not PD, using L-BFGS\n');
   d = lbfgs(-g,old_dirs,old_stps,Hdiag);
end
