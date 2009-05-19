%% MVN Height Weight Demo
plotName = {'heightWeightScatter', 'heightWeightScatterCov'};
for i=2 % 1:2
    if(i==1)
        plotCov = false;
    else
        plotCov=true;
    end
    % Make 2D scatter plot and superimpose Gaussian fit
    rawdata = dlmread('heightWeightDataSimple.txt'); % comma delimited file
    data.Y = rawdata(:,1); % 1=male, 2=female
    data.X = [rawdata(:,2) rawdata(:,3)]; % height, weight
    maleNdx = find(data.Y == 1);
    femaleNdx = find(data.Y == 2);
    classNdx = {maleNdx, femaleNdx};
    figure;
    colors = 'br';
    sym = 'xo';
    for c=1:2
        str = sprintf('%s%s', sym(c), colors(c));
        X = data.X(classNdx{c},:);
        h=scatter(X(:,1), X(:,2), 100, str); %set(h, 'markersize', 10);
        hold on
        if plotCov
            pgauss = fit(MvnDist(), 'data',X);
            gaussPlot2d(pgauss.mu, pgauss.Sigma);
            L = logprob(pgauss, X);
            range1 = [min(data.X(:,1)), max(data.X(:,1))];
            range2 = [min(data.X(:,2)), max(data.X(:,2))];
            %figure;
            %ezsurf(@(x,y)exp(logprob(pgauss,[x y])), range1, range2);
            %ezcontour(@(x,y)exp(logprob(pgauss,[x y])), range1, range2);
        end
    end
    xlabel('height')
    ylabel('weight')
    title('red = female, blue=male');
    if doPrintPmtk, printPmtkFigures(plotName{i}); end;
end


