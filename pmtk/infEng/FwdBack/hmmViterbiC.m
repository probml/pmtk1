% Compiled C code - see hmmViterbiC.c 
% The Matlab version is hmmViterbi.m, written by Kevin Murphy. To use the Matlab
% version, edit FwdBackInfEng.m and change  
%   [path,j,j] = hmmViterbiC(log(eng.pi+eps),log(eng.A+eps),log(eng.B+eps));
% to
%   path = hmmViterbi(eng.pi,eng.A,eng.B);
%
%#author Guillaume Alain 
%#url http://www.cs.ubc.ca/~gyomalin/