classdef SharedParam < dynamicprops
% Objects of this class can be used to store parameters that are shared, (tied) 
% across models. Doing so ensures that the shared data is stored only once and
% that changes made in one place are reflected in the other. Mathematical
% operations return double values not SharedParam instances. 
%
% example:
%
% s = SharedParam(randpd(5),'sigma');
% q = s;
% s(1,1) = 1;
% assert(q(1,1) == 1);

    properties
        value;
        name;
    end
    
    methods
        function obj = SharedParam(value,name)
            if(nargin >= 1)
                obj.value = value;
            end
            if(nargin == 2)
                obj.name = name;
            end
        end
        
        function obj = subsasgn(obj, S, value)
            if(strcmp(S(1).type,'.') && (strcmp(S(1).subs,'value')|| strcmp(S(1).subs,'name')))
                obj = builtin('subsasgn',obj,S,value);
            else
                obj.value = builtin('subsasgn',obj.value,S,value);
            end
        end
        
        function B = subsref(obj, S)
            if(strcmp(S(1).type,'.') && (strcmp(S(1).subs,'value')|| strcmp(S(1).subs,'name')))
                B = builtin('subsref',obj,S);
            else
                B = builtin('subsref',obj.value,S);
            end
        end
        
        function p = mtimes(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            p = mtimes(obj.value,b);
        end
        
        function p = times(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            p = times(obj.value,b);
        end
        
        function vt = transpose(obj)
            vt = obj.value';
        end
        
        function vt = ctranspose(obj)
            vt = obj.value';
        end
        
        function s = plus(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            s = obj.value + b;
        end
        
        function d = minus(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            d = obj.value - b;
        end
        
        function display(obj)
            fprintf('%s* = \n',inputname(1));
            disp(obj.value);
        end
        
        function c = vertcat(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            c = vertcat(obj.value,b);
        end
        
        function c = horzcat(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            c = horzcat(obj.value,b);
        end
        
        function s = sum(obj,varargin)
            s =  sum(obj.value,varargin{:});
        end
        
        function v = rdivide(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            v = rdivide(obj.value,b);
        end
        
        function v = ldivide(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            v = ldivide(obj.value,b);
            
        end
        
        function v = mrdivide(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            v = mrdivide(obj.value,b);
        end
        
        function v = mldivide(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            v = mldivide(obj.value,b);
        end
        
        function v = power(obj,p)
            if(isa(p,'SharedParam'))
                p = p.value;
            end
            v = power(obj.value,p);
        end
        
        function v = mpower(obj,p)
            if(isa(p,'SharedParam'))
                p = p.value;
            end
            v = mpower(obj.value,p);
        end
        
        function v = uminus(obj)
            v = -obj.value;
        end
        
        function v = uplus(obj)
            v = +obj.value;
        end
        
        function v = prod(obj,varargin)
            v = prod(obj.value,varargin{:});
        end
        
        function v = colon(obj,varargin) 
           v = colon(obj.value,varargin{:}); 
        end
        
        function v = svd(obj)
           v = svd(obj.value); 
        end
        
        function v = lt(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            v = lt(obj.value,b);
        end
        
        function v = gt(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            v = gt(obj.value,b);
        end
        
        function v = le(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            v = le(obj.value,b);
        end
        
        function v = ge(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            v = ge(obj.value,b);
        end
        
        function v = ne(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            v = ne(obj.value,b);
        end
        
        function v = eq(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            v = eq(obj.value,b);
        end
        
        function v = and(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            v = and(obj.value,b);
        end
        
        function v = or(obj,b)
            if(isa(b,'SharedParam'))
                b = b.value;
            end
            v = or(obj.value,b);
        end
        
        function v = not(obj)
            v = ~obj.value;
        end
        
        function v = bsxfun(f,obj,val)
           v = bsxfun(f,obj.value,val); 
        end
        
        function [val,ndx] = max(obj,varargin)
           [val,ndx] = max(obj.value,varargin{:}); 
        end
        
        function [val,ndx] = min(obj,varargin)
           [val,ndx] = min(obj.value,varargin{:}); 
        end
        
        function c = copy(obj)
           c = SharedParam(obj.value,obj.name); 
        end
        
        function v = diag(obj,varargin)
           v = diag(obj.value,varargin{:});
        end
        
        function v = reshape(obj,varargin)
           v = reshape(obj.value,varargin{:}); 
        end
        
        function [R,p] = chol(obj,varargin)
           [R,p] = chol(obj.value,varargin{:}); 
        end
        

    end
    
    
    
    
    
    
end

