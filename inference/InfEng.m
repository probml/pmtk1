classdef InfEng
  % inference engine - abstract class
 
  methods(Abstract = true)
      
      [eng, logZ, other] = condition(eng, model, visVars, visVals) 
         % changes state of engine to incorporate obsvered values
         % logZ is probability of evidence or [] if it cnanot be computed.
         % 'other' are other return arguments (engine specific)
         % eg GibbsInfEng returns convergence diagnostics
         % Most engines set other=[].
         
         [postQuery] = marginal (eng, queryVars) 
         % marginalize onto queryVars
  end
 
end