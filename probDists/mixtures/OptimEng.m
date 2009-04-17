classdef OptimEng
  % optimization engine - abstract class
 
  methods(Abstract = true)
     [model, objectiveVal, niter] = fit(model, data);
  end
 
end