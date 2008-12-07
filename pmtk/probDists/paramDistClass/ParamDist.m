classdef ParamDist < ProbDist
   
    
    properties(GetAccess = 'public', SetAccess = 'protected')
        params;
    end
    
    methods
      
         function p = getParams(obj,name)
           if(nargin < 1)
               p = obj.params;
           else
               try
                    p = eval(['obj.params',name]);
               catch
                    p = eval(['obj.',name]);
               end
           end
         end
        
         function v = subsref(obj,S)
         % This checks for the ambiguous case where the user types say
         % obj.params.mu when a joint distribution exists over both mu and
         % Sigma. Does the user want the marginal on mu or the mu field of the
         % joint distribution? If the usage is unambiguous, e.g. m.params.k then
         % no error is issued. 
            
            callbuiltin = false;
            if(numel(S) > 1 && isequal(S(1).type,'.') && isequal(S(2).type,'.') && strcmp(S(1).subs,'params') && ~isa(obj.params,'ProductDist'))
                fromprop = ismember(S(2).subs,properties(obj));
                toprop   = ismember(S(2).subs,properties(obj.params));
                if(fromprop && toprop)
                    error('ParamDist:ambiguousUsage','Ambiguous Usage: Did you want the %s property of the %s joint distribution or the marginal distribution on %s? To obtain the former, type getParams(%s.params,''%s''). To get the latter, type marginal(%s.params,''%s'').',S(2).subs,class(obj.params),S(2).subs,inputname(1),S(2).subs,inputname(1),S(2).subs);
                elseif(fromprop && numel(S) == 2)
                    v = marginal(obj.params,S(2).subs);
                elseif(toprop)
                    v = getParams(obj.params,S(2).subs);
                else
                    callbuiltin = true;
                end
            else
                callbuiltin = true;
            end
            if(callbuiltin)
                  if(numel(S) > 1)
                        v = subsref(builtin('subsref',obj,S(1)),S(2:end)); % peel off one layer to allow for dynamic dispatch
                  else
                        v = builtin('subsref',obj,S);
                  end
            
            end
         end
         
         function obj = set.params(obj,val)
            if(isnumeric(val))
                val = ConstDist(val);
            end
            obj.params = val;
         end
         
        
        
    end
    
end

