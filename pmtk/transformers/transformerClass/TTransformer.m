classdef Transformer % abstract class
  
  properties
    dummy;
  end

  %%  Main methods
  methods
    function obj = Transformer()
      obj.dummy = 1;
    end
    
    function p = addOffset(obj)
      p = false;
    end
    
  end

end