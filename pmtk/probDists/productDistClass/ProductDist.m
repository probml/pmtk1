classdef ProductDist < ProbDist
% This class represents a product of independent probability distributions. For
% efficiency purposes, subclasses will restrict these distributions to be of the
% same type and vectorize many of the operations. While mathematically, the
% order of these distributions is irrelevant, for indexing purposes, the order
% is important and maintained. 
%
%  SUBCLASSES should override: marginal(), mtimes(), setDist()

    properties
        ndistributions = 0;     % The number of distributions
        map;        % an optional map allowing named indexing into the this 
                    % this distribution. 
    end
    
    properties(GetAccess = 'protected', SetAccess = 'protected')
       distArray;  % subclasses should store the n distributions in a 'vectorized' 
                   % way using a single parameter bank, rather than as n objects
                   % storing their own parameters, as done here for generality. 
    end
    
    methods
        
        function obj = ProductDist(distributions,names)
        % Construct a ProductDist from a cell array of distributions.     
        %
        % INPUT: 
        %      distributions - a cell array of probability distributions
        %      names         - an optional cell array of names for these dists.
        %
        % OUTPUT: 
        %      obj           - the constructed ProductDist object
        %      
            if(nargin > 0)
                if(~iscell(distributions))
                    distributions = {distributions};
                end
               
                for i=1:numel(distributions)
                   if(~isa(distributions{i},'ProbDist'))
                       error('A ProductDist can only represent a product of probability distributions. Component %d, of type %s, does not inherit from the ProbDist superclass.',i,class(distributions{i}));
                   end
                end
                obj.distArray = colvec(distributions);
                obj.ndistributions = numel(distributions);
            end
            map = struct;
            if(nargin > 1)
               if(numel(names) ~= numel(distributions))
                   error('The number of names must equal the number of distributions');
               end
               for i=1:numel(names)
                  map.(names{i}) = i; 
               end
            end
            obj.map = map; 
        end
        
        function bool = allConst(obj)
        % return true if all of the distributions are constant
            bool = false;
            for i=1:obj.ndistributions
               if(~isa(marginal(obj,i),'ConstDist')),return;end
            end
            bool = true;
        end   
       
        function d = ndimensions(obj)
        % d is a vector storing the dimensionality of each component distribution. 
            d = zeros(obj.ndistributions,1);
            for i=1:obj.ndistributions
                d(i) = ndimensions(marginal(obj,i));
            end
        end
        
        function logp = logprob(obj,X)
        % evaluate the log probability of each row of X under each distribution.
        % logp is of size size(X,1)-by-ndistributions.
            logp = zeros(size(X,1),obj.ndistributions);
            for i=1:obj.ndistributions
                logp(:,i) = colvec(logprob(marginal(obj,i),X));
            end
        end
        
        function dist = marginal(obj,ndx)
        % Extract the the distribution at position ndx - ndx may also be the
        % name of the distribution. Alternatively, you can use the following
        % syntax obj(ndx). 
            if(iscellstr(ndx)|| ischar(ndx))
                if(numel(fieldnames(obj.map))==0)
                   error('The marginal distributions have not been named'); 
                end
                if(numel(ndx) == 1 || ischar(ndx))
                   dist = obj.distArray{obj.map.(ndx)}; 
                else
                   dist = ProductDist();
                   for i=1:numel(ndx)
                      dist = dist*ProductDist(obj.distArray(obj.map(ndx{i})),sub(fieldnames(obj.map),i));
                   end
                end
            else
                if(numel(ndx) == 1)
                    dist = obj.distArray{ndx};
                else
                    dist = ProductDist(obj.distArray(ndx));
                end
            end
        end
        
        function [c,names] = prod2cell(obj)
        % return all of the distributions as a cell array.
            c = cell(obj.ndistributions,1);
            for i=1:obj.ndistributions
               c{i} = marginal(obj,i);
            end
            names = fieldnames(obj.map);
        end
        
        function c = copy(obj,n,m)
           if(nargin < 3), m = n;end 
           c = cell(n,m);
           for i=1:n
               for j=1:m
                   c{i,j} = obj;
               end
           end
            
        end
        
        function obj = setNames(obj,names)
        % set all of the names of the component distributions. The order of the 
        % names in cell array must correspond to the order of the distributions.
            newmap = struct;
            oldnames = fieldnames(obj.map);
            for i=1:obj.ndistributions
               newmap.(names{i}) = obj.map.(oldnames{i});
            end
            obj.map = newmap;
        end
        
        function obj = rename(obj,oldname,newname)
        % rename a single distribution.    
            val = obj.map.(oldname);
            obj.map = rmfield(obj.map,oldname);
            obj.map.(newname) = val;
        end
        
        function s = prod2struct(obj)
        % return all of the distributions as a struct - only supported if the 
        % distributions are named. 
            s = obj.map;
            names = fieldnames(s);
            for i=1:obj.ndistributions
              s.(names{i}) = marginal(obj,names{i});
            end
        end
        
        function dist = subsref(obj,S)
            % Syntactic sugar for marginal
            
            subs = S(1).subs;
            if(iscell(subs))
                subs = subs{1};
            end
            if(ischar(subs) && ismember(subs,properties(obj)))
                dist = builtin('subsref',obj,S);
                return;
            end
            switch S(1).type
                case {'()','{}'}
                    
                    dist = marginal(obj,subs);
                otherwise % '.'
                    try
                        dist = marginal(obj,subs);
                    catch
                        try
                            if(numel(S) == 1)
                                dist = builtin('subsref',obj,S);
                                return;
                            end
                        catch ME
                            rethrow(ME)
                        end
                    end
            end
            if(numel(S) > 1)
               try
                  dist = subsref(dist,S(2:end));
               catch
                  dist = builtin('subsref',obj,S);
               end
            end
        end
        
%         function obj = subsasgn(obj, S, value)
%         % Allows direct assignment of distributions, e.g. obj(1) = MvnDist    
%            name = S(1).subs;
%            if(iscell(name))
%                name = name{:};
%            end
%            if(numel(S) == 1)
%                if(ischar(name) && ismember(name,properties(obj)))
%                    obj = builtin('subsasgn',obj,S,value);
%                else
%                    if(~isa(value,'ProbDist'))
%                        error('ProductDist only support probability distributions. You are trying to assign a value of class %s',class(value));
%                    else
%                        obj = setDist(obj,name,value);
%                    end
%                end
%            else
%                try
%                    obj = builtin('subsasgn',obj,S,value);
%                catch
%                     dist = marginal(obj,name);
%                     obj = builtin('subsasgn',dist,S(2:end),value);             
%                end
%            end
%            
%         end
        
        function display(obj)
        % custom display of the object    
           disp(obj);
           names = fieldnames(obj.map);
           if(numel(names) > 0 && numel(names) < 10)
               fprintf('\nDistribution Names:\n');
               disp(names');
           end
        end
        
        function prodDist = mtimes(obj1,obj2)
        % Allows Distributions to be combined together via the * operator, e.g. 
        % p = BetaDist() * BernoulliDist(). If two ProductDists are multiplied,
        % no nesting occurs, the result is a new ProductDist with the
        % distributions from both. 
            if(isa(obj1,'ProductDist'))
               if(isa(obj2,'ProductDist'))
                   f1 = fieldnames(obj1.map);
                   f2 = fieldnames(obj2.map);
                   if(numel(intersect(f1,f2)) > 0)
                      error('Each distribution must have a unique name and there is overlap between the two product distributions you are trying to multiply - change the names first with the setNames() method.'); 
                   end
                  prodDist = ProductDist(vertcat(prod2cell(obj1),prod2cell(obj2)),[f1,f2]); 
               else
                  prodDist = ProductDist(vertcat(prod2cell(obj1),{obj2}));
                  prodDist.map = struct;
               end
            else
                  prodDist = ProductDist(vertcat({obj1},prod2cell(obj2)));
                  prodDist.map = struct;
            end
        end
        
        

    end
    
    methods(Access = 'protected')
       
        function obj = setDist(obj,name,newDist)
        % This should be overridden by subclasses to allow easy assignment of a 
        % portion of the parameter bank corresponding to the name. 
            if(ischar(name))
                obj.distArray{obj.map.(name)} = newDist;
            else
                obj.distArray{name} = newDist;
            end
        end
    end
   
   
  
  
    
end

