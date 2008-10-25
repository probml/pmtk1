
nVars = 5;

H = randn(nVars);
H = H'*H;
a = randn(nVars,1);
b = randn;

x = randn(nVars,1);

[f,g] = quadraticLoss(x,H,a,b)

% Finite-Differencing
[f,gFiniteDif] = autoGrad(x,0,@quadraticLoss,H,a,b);

% Complex Purturbations
[f,gComplex] = autoGrad(x,1,@quadraticLoss,H,a,b);

% Simultaneous Purturbation
[f,gSPSA] = SPSA(x,@quadraticLoss,H,a,b);

[g gFiniteDif gComplex gSPSA]