classdef InfEng
  % inference engine - abstract class
 
  methods(Abstract = true)
      [postQuery] = marginal (eng, queryVars)               % marginalize onto queryVars
      [eng, logZ, other] = condition(eng, model, visVars, visVals) % condition on evidence
  end
 
end