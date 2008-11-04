classdef deltaDist < probDist
% This is a degenerate delta distribution centered on a specified
% n-dimensional point. 

    properties
        point;  % This distribution is centered at this possibly n-dimensional point. 
    end

    methods
        
        function obj = deltaDist(point)
        % Constructor    
            obj.point = point; 
        end
        
        function m = mean(obj)
           m = obj.point; 
        end
        
        function m = mode(obj)
           m = obj.point; 
        end
        
        function v = variance(obj)
           v = 0; 
        end


        function h = plot(obj)
            stem(obj.point,'LineWidth',3);
            xlabel('dimension');
            title('delta distribution');
            grid on;
        end
        
        
        
    end
    
    
    methods(Static = true)
        
        function testClass()
           d = deltaDist(5*randn(10,1));
           d.mean
           d.mode
           d.variance
           plot(d);
        end
        
    end
    
    
    
    
end