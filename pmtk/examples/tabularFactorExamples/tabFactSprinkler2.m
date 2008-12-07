%% Sprinkler Example 2
%#broken
[G, CPD, joint, nodes] = makeSprinklerBN;
% joint(c,s,r,w)
C = nodes.C; R = nodes.R; S = nodes.S; W = nodes.W;
false = 1; true = 2;
pSandW = sumv(joint(:,true,:,true), [C R]); % 0.2781
pW = sumv(joint(:,:,:,true), [C S R]); % 0.6471
pSgivenW = pSandW / pW;  % 0.4298

% marginalize then condition
T = TabularFactor(joint);
T1 = marginalize(T, [S W]);
%T2 = enterEvidence(T1, W, true);
T2 = slice(T1, W, true);
T2 = normalizeFactor(T2);
assert(approxeq(pSgivenW, T2.T(true)))

% condition then marginalize
T3 = slice(T, W, true);
T4 = marginalize(T3, S);
T4 = normalizeFactor(T4);

% Another example
pCSRW = sumv(joint(true,true,true,true), []);
pCS = sumv(joint(true,true,:,:), [R,W]);
pRWgivenCS = pCSRW / pCS;

T1 = slice(T, [C S], [true true]);
T2 = marginalize(T1, [R W]);
T2 = normalizeFactor(T2);
assert(approxeq(pRWgivenCS, T2.T(true,true)))