%% MVT Plot Demo
%#testPMTK
useLog = true;
figure;
plot(MvtDist(0.1, [0 0], 0.5*eye(2)), 'useLog', useLog, 'xrange', 2*[-1 1 -1 1], 'useContour', false);
title('mvt(0.1, 0, 0.5 I)');
if useLog, zlabel('log density'); else ylabel('density'); end
if doPrintPmtk, printPmtkFigures('multiT'); end;
figure;
plot(MvnDist([0 0], 0.5*eye(2)), 'useLog', useLog, 'xrange', 2*[-1 1 -1 1], 'useContour', false);
title('mvn(0, 0.5 I)');
if useLog, zlabel('log density'); else ylabel('density'); end
if doPrintPmtk, printPmtkFigures('multiGauss'); end;

