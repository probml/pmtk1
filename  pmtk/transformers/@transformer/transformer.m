classdef transformer % abstract class
  
  properties
    dummy;
  end

  %%  Main methods
  methods
    function obj = transformer()
      obj.dummy = 1;
    end
    
    function p = addOffset(obj)
      p = false;
    end
    
  end

end