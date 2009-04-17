function nestedSamplingDemoLighthouse()
% See Sivia, "Data analysis: a Bayesian tutorial" (2006) p186
% www.inference.phy.cam.ac.uk/bayesys/

%seed = 0; rand('state', seed); randn('state', seed);
[logZ, samples] = nestedSampling(@priorFn, @exploreFn, 100, 1000);
logZ

w = exp([samples.logweight] - logZ);
sx = [samples.x]; sy = [samples.y];
x = sum(w .* sx);
xx = sum(w .* sx.^2);
y = sum(w .* sy);
yy = sum(w .* sy.^2);
fprintf('mean(x)=%6.3f, sd = %6.3f\n', x, sqrt(xx-x*x));
fprintf('mean(y)=%6.3f, sd = %6.3f\n', y, sqrt(yy-y*y));

figure(1); clf
%ndx = {1:20, 101:120, 201:220, 301:320};
ndx = {1:100, 200:300, 300:400, 600:700};
for i=1:4
  subplot(2,2,i)
  plot(sx(ndx{i}), sy(ndx{i}), '.')
  title(sprintf('%d to %d', ndx{i}(1), ndx{i}(end)))
  axis([-2 2 0 2])
end

weights = w;

w = [0.1 0.05 0.1 0.2 0.05 0.3 0.1 0.1];
N = 4;
u = rand;
S = u+N*cumsum(w);

keyboard

Nsamp = 200;
ndx2 = resamplingResidual(1:Nsamp, w);
figure(2); clf
plot(sx(ndx2), sy(ndx2), '.')



function Obj = priorFn()
% sample from uniform prior
Obj.u = rand;
Obj.v = rand;
Obj.x = 4.0 * Obj.u - 2.0;
Obj.y = 2.0 * Obj.v;
Obj.logL = loglik(Obj.x, Obj.y);
Obj.logweight = 0; % dummy

function logL = loglik(x, y)

% data
D = [4.73,  0.45, -1.73,  1.09,  2.19,  0.12, ...
  1.31,  1.00,  1.32,  1.07,  0.86, -0.49, -2.59,  1.73,  2.11, ...
  1.61,  4.98,  1.71,  2.23,-57.20,  0.96,  1.25, -1.56,  2.45, ...
  1.19,  2.17,-10.66,  1.91, -4.16,  1.92,  0.10,  1.98, -2.51, ...
  5.55, -0.47,  1.91,  0.95, -0.78, -0.84,  1.72, -0.01,  1.48, ...
  2.70,  1.21,  4.41, -4.79,  1.33,  0.81,  0.20,  1.58,  1.29, ...
 16.19,  2.75, -2.38, -1.79,  6.50,-18.53,  0.72,  0.94,  3.64, ...
  1.94, -0.11,  1.57,  0.57];

N = length(D);
numer = y/pi;
denom = (D-x).^2 + y^2;
logL = sum(log(numer./denom));


function obj = exploreFn(obj, logLstar)
% sample new object subject to loglik(new) > logLstar
step = 0.1;
naccept = 0; nreject = 0;
for m=1:20
  new.u = obj.u + ((2*rand)-1)*step; 
  %new.u = min(new.u, 1); new.u = max(new.u, 0);
  new.u = new.u - floor(new.u);
  new.v = obj.v + ((2*rand)-1)*step; 
  %new.v = min(new.v, 1); new.v = max(new.v, 0);
  new.v = new.v - floor(new.v);
  new.x = 4*new.u - 2;
  new.y = 2*new.v;
  new.logL = loglik(new.x, new.y);
  new.logweight = 0; % dummy value
  if new.logL > logLstar
    obj = new;
    naccept = naccept + 1;
  else
    nreject = nreject + 1;
  end
  % refine step size to let acceptrance rate converge around 50%
  if (naccept > nreject)
    step = step * exp(1/naccept);
  elseif naccept < nreject
    step = step / exp(1/nreject);
  end
  %fprintf('m=%d, step=%5.3f, accept=%d, reject=%d, arate=%10.5f\n', ...
  %	  m, step, naccept, nreject, naccept/(naccept+nreject));
end




