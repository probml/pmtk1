function [w,fEvals] = L1General(optMethod, gradFunc,w,lambda,params,varargin)
%
% computes argmin_w: gradFunc(w,varargin) + sum lambda.*abs(w)

switch lower(optMethod)
  case 'iteratedridge'
    optfunc = @L1GeneralIteratedRidge;
  case 'projection'
    options.order = -1;
    optfunc = @L1GeneralProjection;
  case 'grafting'
    optfunc = @L1GeneralGrafting;
  case 'orthantwise'
    optfunc = @L1GeneralOrthantWist;
  case 'pdlb'
    optfunc = @L1GeneralPrimalDualLogBarrier;
  case 'sequentialqp'
    optfunc = @L1GeneralSequentialQuadraticProgramming;
  case 'subgradient'
    optfunc = @L1GeneralSubGradient;
  case 'unconstrainedapx'
    optfunc = @L1GeneralUnconstrainedApx;
  case 'unconstrainedapxsub'
    optfunc = @L1GeneralUnconstrainedApx_sub;
  otherwise
    options.order = -1;
    optfunc = @L1GeneralProjection;
end
% The setting of options above is ignored
[w,fEvals] = optfunc(gradFunc,w,lambda,params,varargin);

end