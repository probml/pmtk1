%% Inference in 'misconception network' (Koller and Friedman p99)
%#testPMTK

model = mkMisconceptionUGM();
model.infEng  = EnumInfEng(); 
%model.infEng  = VarElimInfEng(); 
%model.infEng  = JtreeInfEng(); 

false = 1; true = 2;
A = 1; B = 2; C = 3; D = 4;
[marg, logZ] = marginal(model, [A,B,C,D]);
T = pmf(marg);
dispjoint(T) % same as fig 4.2, but different order
assert(approxeq(T(false, true, true, true), 6.9e-5))
Z = exp(logZ);
assert(approxeq(Z, 7201840))
