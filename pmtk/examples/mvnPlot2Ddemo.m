%% Plot an MVN in 2D
%#testPMTK
mu = [1 0]';  S  = [2 1.8; 1.8 2]
figure; plot(MvnDist(mu,S), 'xrange', [-6 6 -6 6], 'useContour', false); title('full');
if doPrintPmtk, printPmtkFigures('gaussPlot2dDemoSurfFull'); end;
figure; plot(MvnDist(mu,S), 'xrange', [-6 6 -6 6], 'useContour', true); title('full');
if doPrintPmtk, printPmtkFigures('gaussPlot2dDemoContourFull'); end;
[U,D] = eig(S);
% Decorrelate
S1 = U'*S*U
figure; plot(MvnDist(mu,S1), 'xrange', [-5 5 -10 10], 'useContour', false); title('diagonal')
if doPrintPmtk, printPmtkFigures('gaussPlot2dDemoSurfDiag'); end;
figure; plot(MvnDist(mu,S1), 'xrange', [-5 5 -10 10], 'useContour', true); title('diagonal');
if doPrintPmtk, printPmtkFigures('gaussPlot2dDemoContourDiag'); end;
% Compute whitening transform:
A = sqrt(inv(D))*U';
mu2 = A*mu;
S2  = A*S*A' % might not be numerically equal to I
assert(approxeq(S2, eye(2)))
S2 = eye(2); % to ensure picture is pretty
% we plot centered on original mu, not shifted mu
figure; plot(MvnDist(mu,S2), 'xrange', [-5 5 -5 5], 'useContour', false); title('spherical');
if doPrintPmtk, printPmtkFigures('gaussPlot2dDemoSurfSpherical'); end;
figure; plot(MvnDist(mu,S2), 'xrange', [-5 5 -5 5], 'useContour', true);
title('spherical');axis('equal');
if doPrintPmtk, printPmtkFigures('gaussPlot2dDemoContourSpherical'); end;