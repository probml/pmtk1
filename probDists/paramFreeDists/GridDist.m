classdef GridDist  < ParamFreeDist

  properties
    density;
    range1;
    range2;
    normconst;
  end


  methods
    function obj = GridDist(logDensityFn, range1, range2)
      % logDensityFn(X(i,:)) is the unnormalized log density at specifed vector
      %   or we can pass in normalized density(i,j) as a vector or matrix.
      % range1 = vector of values on which x1 is evaluated
      % range2 (optional) = vector of values on which x2 is 
      if(nargin == 0), return; end 
      if nargin < 3, range2 = []; end
      obj.range1 = range1;
      obj.range2 = range2;
      densityFn = @(X) exp(logDensityFn(X));
      if isa(logDensityFn, 'function_handle')
        [p, cellsize] = GridDist.evaluateOnGrid(densityFn, range1, range2);
        Z = sum(p(:))*prod(cellsize);
      else
        p = logDensityFn;
        Z = []; % not sure how to compute this...
      end
      obj.normconst = Z;
      obj.density = p/sum(p(:));
    end
    
    function logZ = lognormconst(obj)
      logZ = log(obj.normconst);
    end
    
    function m = marginal(obj, dim)
      if ndimensions(obj) == 1
        error('can only compute marginal on 2d distributions')
      end
      if dim==1
        %foo = @(x1) sum(GridDist.evaluateOnGrid(obj.densityFn, x1, obj.range2),1);
        %m = GridDist(foo, obj.range1);
        p = sum(obj.density,1);
        m = GridDist(p, obj.range1);
      else
        p = sum(obj.density,2);
        m = GridDist(p, obj.range2);
      end
    end
    
    function m = moment(obj, pow)
      if ndimensions(obj)==1
        m = sum(obj.density(:) .* obj.range1(:).^pow);
      else
        [xs, ys] = meshgrid(obj.range1, obj.range2);
        xy = ([xs(:) ys(:)]).^pow;
        p = repmat(obj.density(:), 1, 2);
        m = sum(xy .* p, 1);
      end
    end
    
     function m = mean(obj)
       m = moment(obj, 1);
     end
    
     function v = var(obj)
       % For 2d, returns marginal variances v = [var(X1); var(X2)]
       m1 = moment(obj, 1);
       m2 = moment(obj, 2);
       v = m2 - m1.^2;
     end
    
    function m = mode(obj)
      if ndimensions(obj) == 1
        [pm, ndx] = max(obj.density);
        m = obj.range1(ndx);
      else
        ndx = argmax(obj.density);
        m = [obj.range1(ndx(2)), obj.range2(ndx(1))];
      end
    end

    function d = ndimensions(obj)
      if isempty(obj.range2)
        d = 1;
      else
        d = 2;
      end
    end
    
   
    
    function p = logprob(obj)
      p = [];
    end
    
    function h=plot(obj, varargin)
      % For 2d distributiosn, there are several plotting options
      % 'type' - {['heatmap'], 'contour'}
      [type] = process_options(varargin, 'type', 'heatmap');
      if ndimensions(obj)==1 
        h=bar(obj.dist);
        return;
      end
      switch type
        case 'heatmap' 
          h=imagesc(obj.density); axis xy
          xt = get(gca,'xtick');
          set(gca, 'xticklabel', obj.range1(xt));
          yt = get(gca,'ytick');
          set(gca, 'yticklabel', obj.range2(yt));
        case 'contour'
          h=contour(obj.range1, obj.range2, obj.density);
        otherwise
          error(['unknown type ' type])
      end
    end
    
  end

 
  
  methods(Static = true)
    function testClass()
      disp('foo')
    end
    
    function [p,cellsize] = evaluateOnGrid(densityFn, range1, range2)
      if ~isempty(range2)
        [xs, ys] = meshgrid(range1, range2);
        xy = [xs(:) ys(:)];
        p = reshape(densityFn(xy), size(xs)); %#ok
        cellsize = [range1(2)-range1(1), range2(2)-range2(1)];%#ok
      else
        p = densityFn(range1);%#ok
        cellsize = (range1(2)-range1(1));%#ok
      end
      % scale density so it sums to 1
      %p=p*cellsize;  % p(x:x+dx, y:y+dy) approx p(x,y) dx dy
    end
  end
  
end