%% Sprinkler Example 1

[G, CPD, joint, nodes] = makeSprinklerBN;
C = nodes.C; R = nodes.R; S = nodes.S; W = nodes.W;
fac{C} = tabularFactor(CPD{C}, [C]);
fac{R} = tabularFactor(CPD{R}, [C R]);
fac{S} = tabularFactor(CPD{S}, [C S]);
fac{W} = tabularFactor(CPD{W}, [S R W]);
J = multiplyFactors(fac{:});
assert(approxeq(joint, J.T));