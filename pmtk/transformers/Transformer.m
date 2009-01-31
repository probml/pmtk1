classdef Transformer % abstract class
  
  

  %%  Main methods
  methods
    
    function p = addOffset(obj)
      p = false; % subclasses which add an offset term, e.g. a column of ones, 
                 % must subclass this method and return true;
    end
    
  end
  
  methods(Abstract = true)
      
      [X, transformer] = train(transformer , X);
       X               = test (transformer , X);
      
  end

end