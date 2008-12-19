%% Rainy Day DGM Example
%  V    G
%   \  /  \
%    v    v
%    R    S
V = 1; G = 2; R = 3; S = 4;
dag = zeros(4,4);
dag(V,R) = 1; dag(G,[R S])=1;
CPD{V} = TabularCPD(normalize(ones(1,2)));
T = zeros(2,2,2);
T(1,1,:) = [0.6 0.4];
T(1,2,:) = [0.3 0.7];
T(2,1,:) = [0.2 0.8];
T(2,2,:) = [0.1 0.9];
CPD{R} = TabularCPD(T);
CPD{G} = TabularCPD(normalize(ones(1,2)));
CPD{S} = TabularCPD(mkStochastic(ones(2,2)));
dgm = DgmDist(dag, 'CPDs', CPD);
X = [1 1 1 1;
    1 1 0 1;
    1 0 0 0];
% compute_counts requires data to be in {1,2,...} so we use X+1
dgm = fit(dgm, X+1, 'clampedCPDs', [0 0 1 0]);
delta = dgm.CPDs{V}.T(1)
alpha = dgm.CPDs{G}.T(1)
beta = dgm.CPDs{S}.T(2,1)
gamma = dgm.CPDs{S}.T(1,1)