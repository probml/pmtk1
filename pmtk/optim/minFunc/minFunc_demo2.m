clear all

nInstances = 100;

b = 10*randn;
f = 10*randn(2,1);
H = zeros(2);
H(1,1) = 10*rand;
H(2,2) = 10*rand;
mx = min(H(1,1),H(2,2));
H(1,2) = max(min(10*randn,mx),-mx); % Ensure that determinant is (+)
H(2,1) = H(1,2);
chol(H); % Check that matrix is PD

%% Find Optimal Minimizer
xMin = -(H\f)/2;
fMin = quadraticLoss(xMin,H,f,b);

%% Plot Data


%% Make traced objective and choose methods to run
global wValues
funObj = @(x)tracedObj(x,@quadraticLoss,H,f,b);

methods = {'sd','csd','cg','bb','newton0','lbfgs','bfgs','newton'};

%% Run Methods
colors = getColorsRGB;
symbols = getSymbols;
for m = 1:length(methods)
    figure(m);
    clf; hold on;
    xDomain = [min(xMin(1),0)-1 max(xMin(1),0)+1];
    yDomain = [min(xMin(2),0)-1 max(xMin(2),0)+1];
    xincrement = (xDomain(2)-xDomain(1))/50;
    yincrement = (yDomain(2)-yDomain(1))/50;
    [row,col] = meshgrid(xDomain(1):xincrement:xDomain(2),yDomain(1):yincrement:yDomain(2));
    z = zeros(numel(row),1);
    for i = 1:numel(row)
        z(i,1) = quadraticLoss([row(i);col(i)],H,f,b);
    end
    z = reshape(z,size(row));
    contour(row,col,z,25);

    options.Method = methods{m};
    wValues = [];
    [w fval exitflag output] = minFunc(funObj,zeros(2,1),options);

    h=plot(wValues(1,:),wValues(2,:),'color',colors(m,:),'Marker',symbols{m});
    legend(h,methods{m});
    pause;
end

