
% Example from Edwards p39
S = [3.023 1.258 1.004;...
  1.258 1.709 0.842;...
  1.004 0.842 1.116];

G = zeros(3,3);
G(1,2)=1; G(2,3)=1; G = mkSymmetric(G);

[precMat, covMat] = ggmIPF(S,G); 
precMatEdwards = [0.477 -0.351 0; -0.351 1.19 -0.703; 0 -0.703 1.426];
assert(approxeq(precMat, precMatEdwards))
  
[precMat2, covMat2] = gaussIPFbroken(S, G);
approxeq(full(precMat2), precMatEdwards) % false

%{
% Timing
d = 10;
setSeed(0);
G = mkSymmetric(rand(d,d)>0.8);
G = setdiag(G,0);
S = randpd(d);
tic
[precMat2, covMat2] = gaussIPF(S, G);
toc
tic
[precMat2, covMat2] = ggmIPF(S, G, 'verbose', true);
toc
%}

% Example from Edwards p48
% UG(~ mechanics*vectors*algebra + algebra*analysis*statistics)
G = zeros(5,5);
me = 1; ve = 2; al= 3; an = 4; st = 5;
G([me,ve,al], [me,ve,al]) = 1;
G([al,an,st], [al,an,st]) = 1;
G = setdiag(G,0);
load marks; X = marks;
C = cov(X);
[precMat, covMat] = ggmIPF(C, G);
cov2cor(precMat)



