function varargout = processArgs(args,varargin)
% Similar to process_options, however, allows for arguments to be passed by
% the user either as name/value pairs, or positionally, or both. 

% This function also provides optional enforcement of required inputs, and 
% optional type checking. Argument names, if used, must start with the '-'
% character and not precede any positional arguments. 
%
%% USAGE:
%
% [out1,out2,...,outN] = processArgs(args,'-name1',default1,'-name2',default2,...'-nameN',defaultN)
% 
% The 'args' input is a cell array and in normal usage is simply the
% varargin cell array from the calling function. It contains 0 to N values
% or 0 to N name/value pairs. It may also contain a combination of
% positional and named arguments as long as no named argument precedes a
% positional one. 
%
% Note, this function is CASE INSENSITIVE. 
%
%% ENFORCING REQUIRED ARGUMENTS
%
% To ensure that certain arguments are passed in, (and error if not), add a
% '*' character to the corresponding name as in
%
% [out1,...] = processArgs(args,'*-name1',default1,...
%
%
% Providing empty values, (e.g. {},[],'') for required arguments also errors
% unless these were explicitly passed as named arguments. 
%% TYPE CHECKING
%
% To enforce that the input type, (class) is the same type as the default
% value, add a '+' character to the corresponding name as in
%
% [out1,...] = processArgs(args,'+-name1',default1,...
%
% '+' and '*' can be combined as in
%
% [out1,...] = processArgs(args,'*+-name1',default1,...
%
% or equivalently
%
% [out1,...] = processArgs(args,'+*-name1',default1,...
%% OTHER CONSIDERATIONS
%
% If the user passes in arguments in positional mode, and uses [], {}, or
% '' as place holders, the default values are used in their place. When the
% user passes in values via name/value pairs, this behavior does not
% occur, the explicit value the user specified, (even if [], {}, '') is
% always used. 
%
%% EXAMPLES
% These are all valid usages. Note that here the first and second arguments are
% required and the types of the second and fourth arguments are checked. 
%
% outerFunction(obj,'-first',1,'-second',MvnDist(),'-third',22,'-fourth',10); 
% outerFunction(obj,'-fourth',3,'-second',MvnDist(),'-first',12);
% outerFunction(obj,1,MvnDist(),3);
% outerFunction(obj,1,MvnDist(),3,[]);                           
% outerFunction(obj,'-first',1,'-second',DiscreteDist(),'-third',[]);
% outerFunction(obj,1,MvnDist(),'-fourth',10);                           
% 
% 
% function [a,b,c,d] = outerFunction(obj,varargin)
%   [a,b,c,d] = processArgs(varargin,'*-first',[],'*+-second',ProbDist(),'-third',18,'+-fourth',23);
% end
    %%
    PREFIX = '-';   % prefix that must precede the names of arguments. 
    REQ    = '*';   % require the argument
    TYPE   = '+';   % check the type of the arg against the default type
%% PROCESS VARARGIN - PASSED BY PROGRAMMER
    if ~iscell(args)                                                               ,error('PROGRAMMER ERROR - you must pass in the user''s arguments in a cell array as in processArgs(varargin,''-name'',val,...)');end
    if isempty(varargin)                                                           ,error('PROGRAMMER ERROR - you have not passed in any name/default pairs to processArgs');  end
    argnames  = varargin(1:2:end);
    maxNargs  = numel(argnames);
    required  = cellfun(@(c)ismember(REQ,c),argnames);
    typecheck = cellfun(@(c)ismember(TYPE,c),argnames);
    if ~iscellstr(argnames)                                                        ,error('PROGRAMMER ERROR - you must pass to processArgs name/default pairs'); end
    argnames  = lower(cellfuncell(@(c)c(c~=REQ & c~=TYPE),argnames));
    defaults = varargin(2:2:end);
    varargout = defaults;
    if mod(numel(varargin),2)                                                      ,error('PROGRAMMER ERROR - you have passed in an odd number of arguments to processArgs, which requires name/default pairs');  end
    if any(cellfun(@isempty,argnames))                                             ,error('PROGRAMMER ERROR - empty-string names are not allowed');end
    if nargout >= 0 && nargout ~= maxNargs                                         ,error('PROGRAMMER ERROR - processArgs requires the same number of output arguments as named/default input pairs'); end
    if ~isempty(PREFIX) && ~all(cellfun(@(c)~isempty(c) && c(1)==PREFIX,argnames)) ,error('PROGRAMMER ERROR - processArgs requires that each argument name begin with the prefix %s',PREFIX); end
    if numel(unique(argnames)) ~= numel(argnames)                                  ,error('PROGRAMMER ERROR - you can not use the same argument name twice');end
%% PROCESS ARGS - PASSED BY USER    
    if numel(args) == 0 
        if any(required)                                                           ,error('The following required arguments were not specified:\n%s',cellString(argnames(required))); 
        else  return;
        end
    end
    if ~isempty(PREFIX)
        userstrings = lower(args(cellfun(@ischar,args)));
        problem = ismember(userstrings,cellfuncell(@(c)c(2:end),argnames));
        if any(problem)
            if sum(problem) == 1
                warning('processArgs:missingPrefix','The specified value ''%s'', matches an argument name, except for a missing prefix %s. It will be interpreted as a value, not a name.',userstrings{problem},PREFIX)
            else
                warning('processArgs:missingPrefix','The following values match an argument name, except for missing prefixes %s:\n\n%s\n\nThey will be interpreted as values, not names.',PREFIX,cellString(userstrings(problem)));
            end
        end
    end
    userArgNamesNDX = find(cellfun(@(c)ischar(c) && ~isempty(c) && c(1)==PREFIX,args));
    if ~isempty(userArgNamesNDX) && ~isequal(userArgNamesNDX,userArgNamesNDX(1):2:numel(args)-1)
        if isempty(PREFIX)
            error('\n(1) every named argument must be followed by its value\n(2) no positional argument may be used after the first named argument\n');
        else
            error('\n(1) every named argument must be followed by its value\n(2) no positional argument may be used after the first named argument\n(3) every argument name must begin with the ''%s'' character\n(4) values cannot be strings beginning with the %s character\n',PREFIX,PREFIX); 
        end
    end
    if ~isempty(userArgNamesNDX) && numel(unique(args(userArgNamesNDX))) ~= numel(userArgNamesNDX)
                                                                                                 error('You have specified the same argument name twice');
    end
    enum = enumerate(argnames);
    argsProvided = false(1,maxNargs);
    if isempty(userArgNamesNDX)
        positionalArgs = args;
    elseif userArgNamesNDX(1) == 1
        positionalArgs = {};
    else
        positionalArgs = args(1:userArgNamesNDX(1)-1); 
    end
    if numel(positionalArgs) + numel(userArgNamesNDX) > maxNargs                                , error('You have specified %d too many arguments to the function',numel(positionalArgs)+numel(userArgNamesNDX)- maxNargs);end
    for i=1:numel(positionalArgs)
        arg = args{i};
        if ~isempty(arg)  % don't overwrite default value if positional arg is empty, i.e. '',{},[]
           argsProvided(i) = true;
           if typecheck(i) && ~isa(args{i},class(defaults{i}))                                  , error('Argument %d must be of type %s',i,class(defaults{i}));  end
           varargout{i} = args{i};
        end
    end
    for i=1:numel(userArgNamesNDX)
        argname = lower(args{userArgNamesNDX(i)});
        argvalue = args{userArgNamesNDX(i)+1};
        if ismember(genvarname(argname),fieldnames(enum))
            posindex = enum.(genvarname(argname));
        else
           error('%s is not a valid argument name',argname); 
        end
        if posindex <= numel(positionalArgs)                                                   , error('You cannot specified an argument positionally, and by name in the same function call.');end
        if typecheck(posindex) && ~isa(argvalue,class(defaults{posindex}))                     , error('Argument %s must be of type %s',argname,class(defaults{posindex})); end
        varargout{posindex} = argvalue;
        argsProvided(posindex) = true;
    end
    if any(~argsProvided & required)                                                           , error('The following required arguments were either not specified, or were given empty values:\n%s',cellString(argnames(~argsProvided & required))); end
    