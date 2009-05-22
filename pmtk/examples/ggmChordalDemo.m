%% Exhaustive Search of the set of decomposable GGMs
% We compare accuracy of the MAP model and the posterior mean

function ggmChordalDemo()

d = 4; graphType = 'chain';
n = 10;
seeds = 1:2;
for i=1:length(seeds)
  [loss(i,:), nll(i,:), names]...
    = helperPostModelsExhaustive('seed', seeds(i),'d', d, 'n', n, 'graphType', graphType);      %#ok
end

if length(seeds)>1
  figure;
  subplot(1,2,1); boxplot(loss, 'labels', names); title('KL loss')
  subplot(1,2,2); boxplot(nll, 'labels', names); title('nll')
  suptitle(sprintf('d=%d, n=%d, graph=%s, ntrials = %d', d, n, graphType, length(seeds)))
end

end

function [loss, nll, names] = helperPostModelsExhaustive(varargin)
    [seed, n, d, graphType] = process_options(varargin,...
        'seed'      , 0,...
        'n'         , 10,...
        'd'         , 4,...
        'graphType' ,'chain');
    
    setSeed(seed);
    Gtrue = UndirectedGraph('type', graphType, 'nnodes', d);
    truth = UgmGaussDist(Gtrue, [], []);
    Atrue = Gtrue.adjMat;
    
    truth = mkRndParams(truth);
    Y = sample(truth, n);
   
    prec{1} = inv(truth.Sigma); names{1} = 'truth';
    prec{2} = inv(cov(Y)); names{2} = 'emp';
   
    [postG, BIC, GGMs, mapG, mapPrec,  meanG, meanPrec] = ...
        computePostAllModelsExhaustive(Y);
      NG = length(GGMs);
    
    prec{3} = meanPrec; names{3} = 'mean';
    prec{4} = mapPrec; names{4} = 'mode';

    % predictive accuracy
    nTest = 1000;
    Ytest = sample(truth, nTest);
    for i=1:4
        SigmaHat = inv(prec{i});
        model = MvnDist(mean(Y), SigmaHat);
        nll(i) = -sum(logprob(model, Ytest),1);
    end

    figure;
    bar(postG);
    trueNdx = [];
    for i=1:length(GGMs)
        if isequal(GGMs{i}.G.adjMat, Atrue)
            trueNdx = i; break;
        end
    end
    if isempty(trueNdx)
        title(sprintf('p(G|D), truth=nondecomp, d=%d, n=%d', n, d));
    else
        title(sprintf('p(G|D), truth=%d, d=%d, n=%d', trueNdx, d, n));
    end

    figure;
    BICn = exp(normalizeLogspace(BIC));
    bar(BICn); title('normalized BIC')
    
    
    figure;
    for i=1:4
        SigmaHat = inv(prec{i});
        M = SigmaHat * inv(truth.Sigma); %#ok
        loss(i) = trace(M) - log(det(M)) - d; %#ok % KL loss 
        subplot(2,2,i);
        %imagesc(prec{i}); colormap('gray');
        hintonDiagram(prec{i});
        title(sprintf('%s, KL = %3.2f, nll =%3.2f', names{i}, loss(i), nll(i)));
        hold on
    end

    [pmax, Gmax] = max(postG);
    Gmap = GGMs{Gmax}.G;
    Amap = Gmap.adjMat;
    %Graphlayout('adjMatrix',Amap,'undirected',true);
    %draw(Gmap); title('Gmap')
    hamming = sum(abs(Amap(:) - Atrue(:)));
    
    figure;
    subplot(2,2,1); hintonDiagram(Atrue); title('true G')
    subplot(2,2,2); hintonDiagram(Amap); title(sprintf('G map, hamdist=%d', hamming))
    subplot(2,2,3); hintonDiagram(meanG); title('post mean G');

    
    drawnow
    restoreSeed;
end

%%
function [postG, BIC, GGMs, mapG, mapPrec, postMeanG, postMeanPrec] = ...
  computePostAllModelsExhaustive(Y)

[n,d] = size(Y);
%Phi = 0.1*eye(d); delta = 5; % hyper-params
Phi = var(Y(:))*eye(d); delta = 3; % hyper-params
GGMs = UgmGaussChordalDist.mkAllGgmDecomposable(delta, Phi);
N = length(GGMs);
prior = normalize(ones(1,N));
logpostG = zeros(1,N);
for i=1:N
  logpostG(i) = log(prior(i)) + logmarglik(GGMs{i}, 'data', Y);
end
postG = exp(normalizeLogspace(logpostG));
 
ll = zeros(1,N); BIC = zeros(1,N);
for i=1:N
  tmp = fit(UgmGaussDist(GGMs{i}.G), Y);
  ll(i) = sum(logprob(tmp, Y),1);
  BIC(i) = ll(i) - dof(tmp)*log(n)/2;
end

bestNdx = argmax(logpostG);
mapG = GGMs{bestNdx}.G;
nstar = n-1; % since mu is unknown
Sy = n*cov(Y,1);
deltaStar = delta + nstar; PhiStar = Phi + Sy;
mapPrec = meanInverse(HiwDist(mapG, deltaStar, PhiStar));

postMeanPrec = zeros(d,d);
postMeanG = zeros(d,d);
% Armstrong thesis p80
for i=1:N
  postMeanPrec = postMeanPrec + postG(i) * meanInverse(HiwDist(GGMs{i}.G, deltaStar, PhiStar));
  postMeanG = postMeanG + postG(i) * GGMs{i}.G.adjMat;
end

end


