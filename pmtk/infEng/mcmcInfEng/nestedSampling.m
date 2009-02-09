function [logZ, samples] = nestedSampling(priorFn, exploreFn, nobj, nsamples)
% nested sampling based on Sivia and Skilling's C code
% www.inference.phy.cam.ac.uk/bayesys/

for i=1:nobj
  obj(i) = feval(priorFn);
end
logwidth = log(1-exp(-1/nobj));
warning off; logZ = log(0); warning on
for k=1:nsamples
  LL = [obj.logL];
  [worstLL, worstObj] = min(LL);
  obj(worstObj).logweight = logwidth + worstLL;
  logZnew = logsumexp2(logZ, obj(worstObj).logweight);
  logZ = logZnew;
  samples(k) = obj(worstObj);

  % sample an object other than the worst (if n>1)
  copy = ceil(nobj * rand()); % 1..nobj
  while ((copy==worstObj) && (nobj>1)) 
    copy = ceil(nobj * rand());
  end
  obj(worstObj) = feval(exploreFn, obj(copy), worstLL);
  logwidth = logwidth - 1/nobj;
  if mod(k,100)==0
    fprintf('k=%d, logwidth=%5.3f, logZ=%5.3f\n',k, logwidth, logZ);
  end
end
  
%%%%

function z = logsumexp2(x, y)
% z = log(exp(x)+exp(y))
if x>y
  z = x+log(1+exp(y-x));
else
  z = y+log(1+exp(x-y));
end
