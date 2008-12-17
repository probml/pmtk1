classdef ParamDist < ProbDist
   
    %{
    properties(GetAccess = 'public', SetAccess = 'protected')
        params;
    end
    %}
  
    methods
        
      function d = ndimensions(obj)
        % By default, we asssume the distribution is over a scalar rv
        % If the class defines a vector rv, it should over-ride this
        % method.
        d = 1;
      end
      
      function d = ndistrib(obj)
        % By default, we asssume the distribution is a single distribution
        % not a product/set.
        d = 1;
      end
      
      
      function Xc = impute(obj, X)
        % Fill in NaN entries of X using posterior mode on each row
        [n] = size(X,1);
        Xc = X;
        for i=1:n
          hidNodes = find(isnan(X(i,:)));
          if isempty(hidNodes), continue, end;
          visNodes = find(~isnan(X(i,:)));
          visValues = X(i,visNodes);
          postH = predict(obj, visNodes, visValues);
          Xc(i,hidNodes) = rowvec(mode(postH));
        end
      end

     %{
          %% Get/Set
          
         function p = getParams(obj,name,pointEstimate)
           if(nargin < 3)
               pointEstimate = '';
           end
           if(nargin < 1)
               p = obj.params;
           else
               try
                    p = eval(['obj.params',name]);
               catch
                    p = eval(['obj.',name]);
               end
               if(strcmpi(pointEstimate,'point'))
                  try
                      m = mode(p);
                  catch
                      return;
                  end
                  p = m;
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
         
         %}
         
%          function display(obj)
%              fprintf('\n%s: \n',class(obj));
%            
%              if(isa(obj.params,'ProductDist'))
%                 modelNames = fieldnames(obj.params.map)';
%                 propNames = setDiff(properties(obj),modelNames);
%              else
%                  modelNames = class(obj.params);
%                  propNames = properties(obj);
%              end
%              fprintf('Properties: \n');
%              disp(propNames);
%              fprintf('\nModel Parameters:\n');
%              disp(modelNames);
%              
%              
%          end
         
        
        
    end
    
end

