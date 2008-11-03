% Example from Edwards p39
S = [3.023 1.258 1.004;...
  1.258 1.709 0.842;...
  1.004 0.842 1.116];
G = zeros(3,3);
G(1,2)=1; G(2,3)=1; G = mkSymmetric(G);
precMat1 = covselPython(S, G)
precMat2 = ggmIPF(S,G)
%precMat3 = covselFastPython(S, G);
precMat3 = covselProj(S, G);
covMat = inv(precMat2);
precMatEdwards = [0.477 -0.351 0; -0.351 1.19 -0.703; 0 -0.703 1.426];
assert(approxeq(precMat1, precMatEdwards))
assert(approxeq(precMat2, precMatEdwards))
assert(approxeq(precMat3, precMatEdwards))


% Marks - Edwards p48
G = zeros(5,5);
me = 1; ve = 2; al= 3; an = 4; st = 5;
G([me,ve,al], [me,ve,al]) = 1;
G([al,an,st], [al,an,st]) = 1;
G = setdiag(G,0);
load marks; X = marks;
S = cov(X);
precMat1 = ggmIPF(S, G)
precMat2 = covselPython(S, G)
precMat3 = covselProj(S, G)

pcorMatEdwards = eye(5,5);
pcorMatEdwards(2,1) = 0.332;
pcorMatEdwards(3,1:2) = [0.235 0.327];
pcorMatEdwards(4,1:3) = [0 0 0.451];
pcorMatEdwards(5,1:4) = [0 0 0.364 0.256];
pcorMatEdwards = mkSymmetric(pcorMatEdwards);

assert(approxeq(pcorMatEdwards, abs(cov2cor(precMat1))))
assert(approxeq(pcorMatEdwards, abs(cov2cor(precMat2))))
assert(approxeq(pcorMatEdwards, abs(cov2cor(precMat3))))

% Timing
d = 50;
G = mkSymmetric(rand(d,d)>0.8);
G = setdiag(G,1);
S = randpd(d);
tic; precMat1 = covselPython(S, G); toc
tic; precMat2 = covselProj(S, G); toc
assert(approxeq(precMat1, precMat2))

