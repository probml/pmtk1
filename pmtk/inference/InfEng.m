classdef InfEng
  % inference engine - abstract class
 
  methods(Abstract = true)
     
      [postQuery,eng] = marginal (eng, queryVars)               % marginalize onto queryVars
      eng             = condition(eng, model, visVars, visVals) % condition on evidence
      samples         = sample   (eng, n)                       % return samples
      
  end
  
  methods
      
      function eng = InfEng()
         eng;  %#ok 
      end
      
  end
 
end