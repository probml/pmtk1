function varargout = processArgs(args,varargin)
% Similar to process_options, however, allows for arguments to be passed by
% the user either as name/value pairs, or positionally. This function also
% provides optional enforcement of required inputs, and optional type
% checking. 
%
%% USAGE:
%
% [out1,out2,...,outN] = processArgs(args,'name1',default1,'name2',default2,...'nameN',defaultN)
% 
% The 'args' input is a cell array and in normal usage is simply the
% varargin cell array from the calling function. It contains 0 to N values
% or 0 to N name/value pairs. 
%
% Note, this function is CASE INSENSITIVE. 
%
%% ENFORCING REQUIRED ARGUMENTS
%
% To ensure that certain arguments are passed in, (and error if not), add a
% '*' character to the corresponding name as in
%
% [out1,...] = processArgs(args,'*name1',default1,...
%
%% TYPE CHECKING
%
% To enforce that the input type, (class) is the same type as the default
% value, add a '+' character to the corresponding name as in
%
% [out1,...] = processArgs(args,'+name1',default1,...
%
% '+' and '*' can be combined as in
%
% [out1,...] = processArgs(args,'*+name1',default1,...
%
%% OTHER CONSIDERATIONS
%
% If the user passes in arguments in positional mode, and uses [], {}, or
% '' as place holders, the default values are used in their place. When the
% user passes in values via name/value pairs, this behavior does not
% occur, the explicit value the user specified, (even if [], {}, '') is
% used. 
%
%% EXAMPLES
% These are all valid usages. Note that here the first and second arguments are
% required and the types of the second and fourth arguments are checked. 
%
% outerFunction('first',1,'second',MvnDist(),'third',22,'fourth',10)
% outerFunction('fourth',3,'second',MvnDist(),'first',12)
% outerFunction(3,MvnDist())
% outerFunction(3,MvnDist(),[],10)                        % default value of 18 used for third argument
% outerFunction('first',1,'second',MvnDist(),'third',[])  % default value of 18 NOT used for third argument. 
% 
% function output = outerFunction(obj,varargin)
%   
%   [a,b,c,d] = processArgs(varargin,'*first',[],'*+second',MvnDist(),'third',18,'+fourth',23)
%     .
%     .
%     .
% end
% 
    if(mod(numel(varargin),2))
        error('Programmer Error - you have passed in an odd number of arguments to processArgs, which requires name/default pairs.');
    end
    
    ispositional = ~(allSameTypes(args(1:2:end-1)) && ischar(args{1}));
    nargsMax = numel(varargin)/2;
    if ispositional
        nargs = numel(args);
        argValues = args;
    else
        nargs = numel(args)/2;
        argValues = args(2:2:end);
        userNames = lower(args(1:2:end-1));
    end
        
    if(nargout ~= nargsMax)
        error('Programmer Error - wrong number of output arguments for processArgs.');
    end
    
    if(nargs > nargsMax)
        error('You have specified too many arguments to a function.');
    end
    
    expectedNames = varargin(1:2:end-1);
    required      = cellfun(@(c)ismember('*',c),expectedNames);
    typeCheck     = cellfun(@(c)ismember('+',c),expectedNames);
    expectedNames = cellfuncell(@(c)c(c~='*'& c~='+'),expectedNames);
    defaults      = varargin(2:2:end);
    varargout     = defaults;
    checkTypes();
    nameMap = createStruct(lower(expectedNames),1:nargsMax);
    
    if(ispositional)
        if(nargs < nargsMax && sum(required(nargs+1:end)))
            if(sum(required) == 1)
                error('Argument %d is required.',sub(1:nargsMax,required));
            else
                error('Arguments %s are required.',num2str(sub(1:nargsMax,required)));
            end
        end
        for i=1:nargs
            if(~isempty(argValues{i}))
                varargout(i) = argValues(i);
            end
        end
    else % named
       stillRequired = required;
       for i=1:nargs
           if(~isfield(nameMap,userNames{i}))
               error('%s is not a valid a valid argument name',userNames{i});
           end
           ndx = nameMap.(userNames{i});
           varargout(ndx) = argValues(i);
           stillRequired(ndx) = false;
       end
        if(any(stillRequired))
            fprintf('The following required arguments were not specified:\n');
            display(expectedNames(stillRequired));
            error('Missing required arguments');
        end
    end

    function checkTypes()
        for j=1:nargs
            if(typeCheck(j))
                if(~isa(argValues{j},class(defaults{j})))
                    if(ispositional)
                        error('Argument %d must be of type %s',j,class(defaults{j}));
                    else
                        error('Argument %s must be of type %s',expectedNames{j},class(defaults{j}));
                    end
                end
            end
        end
    end
 
end