%% Current Constructor
function m = MvnDist(mu, Sigma,  varargin)
    if nargin == 0
        mu = []; Sigma = [];
    end
    [m.domain, m.prior, m.fitMethod, m.fitArgs, m.infEng] = process_options(varargin, ...
        'domain', 1:numel(mu), 'prior', 'none', 'fitMethod', 'mle', ...
        'fitArgs', {}, 'infEng', GaussInfEng());
    m.mu = mu; m.Sigma = Sigma;
end

%% Proposed Constructor
function m = MvnDist(varargin)
    
    [m.mu,m.Sigma, m.domain, m.prior, m.fitMethod, m.fitArgs, m.infEng] = processArgs(varargin, ...
        '-mu',[],'-Sigma',[],'-domain', [], '-prior', 'none', '-fitMethod', 'mle', ...
        '-fitArgs', {}, '-infEng', GaussInfEng());
     if(isempty(m.domain))
         m.domain = 1:numel(mu);
     end
end


%% Ways in which the proposed constructor could be called

MvnDist('mu',[1,0],'Sigma',randpd(2),'fitMethod','map')
MvnDist([1,0],randpd(2))
MvnDist([1,0,randpd(2),[],[],'map')  % default values used for 3rd, 4th, and unspecified inputs


%% Proposed Constructor with enforced parameters and type checking. 
% By adding stars to the mu and Sigma names, we tell processArgs to error
% if they are not specified, (either positionally or by name). By adding
% '+' symbols to mu, Sigma, domain, fitMethod, fitArgs, we ware telling
% processArgs to error if the user supplies values of types different than
% the default value types for these arguments, i.e. we are enforcing that
% mu, Sigma, domain are of type 'double' and that 'fitMethod' is of type 'char'
% and fitArgs of type cell. 

function m = MvnDist(varargin)
    
    [m.mu,m.Sigma, m.domain, m.prior, m.fitMethod, m.fitArgs, m.infEng] = processArgs(varargin, ...
        '*+-mu',[],'*+-Sigma',[],'+-domain', [], '-prior', 'none', '+-fitMethod', 'mle', ...
        '+-fitArgs', {}, '-infEng', GaussInfEng());
     if(isempty(m.domain))
         m.domain = 1:numel(mu);
     end
end


 



