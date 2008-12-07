%% Sprinkler Example 1
%#broken
[G, CPD, joint, nodes] = makeSprinklerBN;
C = nodes.C; R = nodes.R; S = nodes.S; W = nodes.W;
fac{C} = TabularFactor(CPD{C}, [C]);
fac{R} = TabularFactor(CPD{R}, [C R]);
fac{S} = TabularFactor(CPD{S}, [C S]);
fac{W} = TabularFactor(CPD{W}, [S R W]);
J = multiplyFactors(fac{:});
assert(approxeq(joint, J.T));