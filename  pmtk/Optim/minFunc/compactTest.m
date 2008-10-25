function [] = compactTest(funObj,x,options,varargin)

nVars = length(x);

[f,g] = funObj(x,varargin{:});
funEvals = 1;


B0 = .01*eye(nVars);

S = zeros(nVars,0);
Y = zeros(nVars,0);
B0i = B0^-1;

B = B0;
for iter = 1:250
    d = -B\g;
    
    if iter > 1
        %B0i-B0i*N*(M + N'*B0i*N)^-1*N'*B0i;
        %dCompact = -(B0i-B0i*N*(-M + N'*B0i*N)^-1*N'*B0i)*g
        dCompact = -B0i*g + B0i*(N*((-M + N'*B0i*N)^-1*(N'*(B0i*g))));
        %pause;
    end
    
    t = 1;
    [f_new,g_new] = funObj(x+t*d,varargin{:});
    funEvals = funEvals+1;
    while f_new > f
        t = t/2;
        [f_new,g_new] = funObj(x+t*d,varargin{:});
    funEvals = funEvals+1;
    end
    
    y = g_new-g;
    s = t*d;
    
    if y'*s > 1e-10
        B = B + (y*y')/(y'*s) - (B*s*s'*B)/(s'*B*s);
        
        % Compact Representation
        S = [S s];
        Y = [Y y];
        k = size(Y,2);
        L = zeros(k);
        for j = 1:k
            for i = j+1:k
                L(i,j) = S(:,i)'*Y(:,j);
            end
        end
        D = diag(diag(S'*Y));
        N = [B0*S Y];
        M = [S'*B0*S L;L' -D];
        Bcompact = B0 - N*M^-1*N';
        
        %sum(abs(B(:)-Bcompact(:)))
        %pause;
        
        % Compute B*d as: B0*d - N*(M^-1*(N'*d))
    end
    
    x = x+t*d;
    f = f_new;
    g = g_new;
    
    fprintf('iter = %d, fEvals = %d, f = %.4f\n',iter,funEvals,f);
    
    if sum(abs(t*d)) < 1e-7
        break;
    end
end