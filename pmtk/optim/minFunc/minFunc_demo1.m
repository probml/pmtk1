clear all
close all
warning off all

nInstances = 1000;
nVars = 500;

X = randn(nInstances,nVars);
w = randn(nVars,1);
y = X*w + randn(nInstances,1);

funObj = @(w)GaussianLoss(w,X,y);

wLS = X\y;
minObj = funObj(wLS);

options.TolFun = 1e-10;
options.TolX = 1e-10;
options.maxFunEvals = 100;
options.HvFunc = @(w,v)GaussianHv(w,v,X,y);

figure(1);clf; hold on
colors = getColorsRGB;
symbols = getSymbols;
methods = {'sd','csd','cg','bb','newton0','lbfgs','bfgs','newton'};
for i = 1:length(methods)
    methods{i}
    options.Method = methods{i};
    [w fval exitflag output] = minFunc(funObj,zeros(nVars,1),options);
    fprintf('Error in objective value: %e\n',abs(funObj(wLS)-funObj(w)));
    fprintf('Maximum error parameter estimate: %e\n',max(abs(wLS-w)));
    fprintf('Number of function evaluations: %d\n',output.funcCount);
    
    figure(1);
        plot(output.trace.funcCount,log(abs(output.trace.fval-minObj)+1e-10),'color',colors(i,:),'Marker',symbols{i});
    legend(methods{1:i});
    xlabel('Function Evaluation');
    ylabel('log( objective value - optimal )');
    xlim([1 options.maxFunEvals]);
    pause;
end
