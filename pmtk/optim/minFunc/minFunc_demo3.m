clear all
close all

nVars = 10;

funObj = @rosenbrock;

minObj = funObj(ones(nVars,1));

options.TolFun = 1e-10;
options.TolX = 1e-10;
options.maxFunEvals = 250;
options.Display = 'Full';

figure(1);clf; hold on
colors = getColorsRGB;
symbols = getSymbols;
methods = {'sd','csd','cg','scg','bb','newton0','newton0lbfgs','lbfgs','bfgs','mnewton','newton','tensor'};
for i = 1:length(methods)
    methods{i}
    options.Method = methods{i};
    [w fval exitflag output] = minFunc(funObj,zeros(nVars,1),options);
    fprintf('Error in objective value: %e\n',abs(minObj-funObj(w)));
    fprintf('Maximum error parameter estimate: %e\n',max(abs(ones(nVars,1)-w)));
    fprintf('Number of function evaluations: %d\n',output.funcCount);
    
    figure(1);
        plot(output.trace.funcCount,log(abs(output.trace.fval-minObj)+1e-10),'color',colors(i,:),'Marker',symbols{i});
    legend(methods{1:i});
    xlabel('Function Evaluation');
    ylabel('log( objective value - optimal )');
    xlim([1 options.maxFunEvals]);
    pause;
end
