

methods = {};
methods{end+1} = 'covselPython';
methods{end+1} = 'covselIpf';
methods{end+1} = 'covselIpfClqs';
% methods{end+1} = 'covselIpfR'; % fails on rnd tests
methods{end+1} = 'covselMinfunc';
% methods{end+1} = 'covselProj'; % buggy


% Example from Edwards p39
S = [3.023 1.258 1.004;...
  1.258 1.709 0.842;...
  1.004 0.842 1.116];
G = zeros(3,3);
G(1,2)=1; G(2,3)=1; G = mkSymmetric(G);
precMatEdwards = [0.477 -0.351 0; -0.351 1.19 -0.703; 0 -0.703 1.426];

%{
% NewsGroups data
    load 20news_w100
    docs = double(documents);
    sigma = full(docs*docs')/16242;
%}


for m=1:length(methods)
  precMat{m} = feval(methods{m}, S, G);
  correct = approxeq(precMatEdwards, precMat{m});
  fprintf('method %s correct %d\n', methods{m}, correct);
end



% Marks - Edwards p48
G = zeros(5,5);
me = 1; ve = 2; al= 3; an = 4; st = 5;
G([me,ve,al], [me,ve,al]) = 1;
G([al,an,st], [al,an,st]) = 1;
G = setdiag(G,0);
load marks; X = marks;
S = cov(X);
pcorMatEdwards = eye(5,5);
pcorMatEdwards(2,1) = 0.332;
pcorMatEdwards(3,1:2) = [0.235 0.327];
pcorMatEdwards(4,1:3) = [0 0 0.451];
pcorMatEdwards(5,1:4) = [0 0 0.364 0.256];
pcorMatEdwards = mkSymmetric(pcorMatEdwards);

for m=1:length(methods)
  precMat{m} = feval(methods{m}, S, G);
  correct = approxeq(pcorMatEdwards, abs(cov2cor(precMat{m})));
  fprintf('method %s correct %d\n', methods{m}, correct);
end




% Timing on random problems 
d = 10;
setSeed(0);
ns = [d*2 d/2];
clear precMat
for trial=1:length(ns)
  n = ns(trial);
  X = randn(n,d);
  S = cov(X) + 0.001*eye(d);
  G = mkSymmetric(rand(d,d)>0.8);
  G = setdiag(G,0);

  fprintf('n=%d, d=%d, S is pd%d\n\n', n, d, isposdef(S));
  
  figure;
  for m=1:length(methods)
    tic;
    precMat{m} = feval(methods{m}, S, G);
    t=toc;
    correct = approxeq(precMat{m}, precMat{1}, 1e-1);
    Ghat = precmatToAdjmat(precMat);
    valid = isequal(G, Ghat);
    fprintf('method %s eq1 %d, valid %d, time %5.3f\n', ...
      methods{m}, correct, valid, t);
    subplot(2,2,m); imagesc(precMat{m}); 
    title(sprintf('n%d, d%d, %s', n, d, methods{m}));
  end
  
  
end

