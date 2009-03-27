classdef LinearDynamicalSystemDist < ParamDist
% A discrete time, stochastic, (switching) linear dynamical system 
% (state space model), with Gaussian noise. 
    
    properties
        % Let z(t) denote the hidden state at time t and y(t) the
        % observation at time t. If specifed, x(t) denotes an additional
        % input, (control) vector for time t.
        
        infMethod;    % the inference method to use: kalman_filter() or kalman_smoother
        startDist;    % An MvnDist() over starting hidden states z(0)
        sysNoise;     % An MvnDist() - the noise model of the system dynamics. 
        obsNoise;     % An MvnDist() - the noise model for the observations.   
        sysMatrix;    % (stateSize-by-stateSize)  
        obsMatrix;    % (obsSize-by-stateSize)    
        % z(t+1) = sysMatrix*z(t) + w(t),  w ~ sysNoise, z(0)~ startDist    
        % y(t)   = obsMatrix*z(t) + v(t),  v ~ obsNoise                   
        
       
        % optional
        inputMatrix;  %  z(t+1) = sysMatrix*z(t) + inputMatrix*x(t) + w(t)  
                      % where an additional input vector x(t) is
                      % conditioned upon. 
                      
        modelSwitch;  % If specified, modelSwitch(t) indicates the index of
                      % model to use at time step t. sysNoise and obsNoise
                      % must then be vectorized to represent
                      % numel(modelSwitch) distributions, and sysMatrix,
                      % obsMatrix,(and if present, inputMatrx) must have a
                      % third dimension so that,e.g.
                      % sysMatrix(:,:,modelSwitch(t)) returns the sysMatrix
                      % to be used at time step t. In a vectorized MvnDist,
                      % mu is d-by-ndistributions and Sigma is
                      % d-by-d-by-ndistributions. 
                      
        obsSize;      % the dimensionality of the observations
        stateSize;    % the dimensionality of the hidden states
    end
    
    methods
        
        function model = LinearDynamicalSystemDist(varargin)
        % Constructor    
            [model.startDist,model.sysNoise ,model.obsNoise     ,  ...
             model.sysMatrix,model.obsMatrix,model.inputMatrix  ,  ...
             model.modelSwitch,model.infMethod,model.obsSize    ,  ...
             model.stateSize]                                 =    ...
             process_options(varargin                           ,  ...
                'startDist'  ,[],'sysNoise'  ,[],'obsNoise'   ,[], ...
                'sysMatrix'  ,[],'obsMatrix' ,[],'inputMatrix',[], ...
                'modelSwitch',[],'infMethod' ,@kalman_smoother   , ...
                'obsSize'    ,[], 'stateSize',[]                );
             
             if ischar(model.infMethod)
               model.infMethod = str2func(model.infMethod);
             end
             
             if ~isempty(model.obsMatrix)
                 [model.obsSize,model.stateSize] = size(model.obsMatrix);
             end
        end
        
        function [model,LL] = fit(model,data,varargin)
        % Fit the model via EM
        %
        % data(:,t,l) is the observation vector at time t for sequence l. 
        % If the sequences are of different lengths, you can pass in a cell
        % array, so data{l} is an obsSize*T matrix, where T is the number
        % of time steps. 
        %
        % You can enforce that the cov matrices of the noise models are
        % diagonal by setting the covType property of the respective
        % MvnDists.
        %
        % To fit in 'ARmode' i.e. a gauss-markov process, set the
        % observation Matrix to the identity matrix and the covariance
        % matrix of obsNoise to be the all zeros matrix. 
        % 
        % The current values for sysMatrix,obsMatrix, startDist, obsNoise,
        % sysNoise, etc are used as initialization values for EM. If these
        % are not specified, random values are used. 
        % 
        % optional inputs:
        %
        % 'maxIter'     - maximum number of EM iterations to perform, [20].
        %
        % 'verbose'     -  if true, the LL is printed after each
        %                  iteration [false]
        %
        % 'constraintFunc' -   if specified, called after every M step to
        %                      enforce constraints. See kalman_filter.m for
        %                      details on the required interface. 
        %
        % 'constraintArgs' -   additional arguments in a cell array passed
        %                      to constraintFunc.
        %  
        
           [maxIter,verbose,constraintFunc,constraintArgs] = ...
               process_options(varargin   ,...
               'maxIter'          ,20     ,...
               'verbose'          ,false  ,...
               'constraintFunc'   ,[]     ,...
               'constraintArgs'   ,{}     );
           
           if iscell(data), model.obsSize = size(data{1},1);
           else             model.obsSize = size(data,1);    end
           
           [model,A, C, Q, R, initz, initV,diagQ,diagR,ARmode] = initParams(model);
           [model.sysMatrix, model.obsMatrix, model.sysNoise.Sigma,...
            model.obsNoise.Sigma, model.startDist.mu, model.startDist.Sigma, LL]...
            = learn_kalman(data, A, C, Q, R, initz, initV, maxIter, diagQ, diagR, ARmode,verbose,constraintFunc,constraintArgs{:});
        end
        
        function [Z,ZZ,loglik] = marginal(model,queryVars, visVars, visVals)
        % Use a Kalman filter,(or smoothing) to decode an observation 
        % sequence, 'Y'. Currently, queryVars must be 'Z', the hidden
        % sequence and you must specify,(condition on) the observation
        % sequence 'Y'. Additionally, you can also condition on an input
        % control sequence 'X' which is used in combination with
        % model.inputMatrix). 
        %
        % The output, Z is a vectorized MvnDist with 
        % Z.mu(:,t) = E[Z(:,t)| Y(:,1:T)] 
        % Z.Sigma(:,:,t) = Cov[Z(:,t) | Y(:,1:T)]
        %
        %        OR
        %
        % Z.mu(:,t) = E[Z(:,t) | Y(:,1:T), X(:, 1:T)]
        % Z.Sigma(:,:,t) = Cov[Z(:,t) | Y(:,1:T), X(:,1:T)]
        % 
        %
        % ZZ is the same as Z except that 
        %
        % ZZ.Sigma(:,:,t) = Cov[Z(:,t), Z(:,t-1) | Y(:,1:T)] t >= 2
        %
        %       OR
        %
        % ZZ.Sigma(:,:,t) = Cov[Z(:,t), Z(:,t-1) | Y(:,1:T), X(:,1:T)] t >= 2
        % 
        % EXAMPLES:
        % Z      = marginal(model,'Z','Y',observations)
        % [Z,ZZ] = marginal(model,'Z',{'Y','X'},{observations,controlSequence})
        %
            if ~ischar(queryVars) || ~strcmpi(queryVars,'z')
               error('Querying vars other than Z not currently supported'); 
            end
            if iscell(visVars)
                visVars = lower(visVars);
                if numel(visVars) == 1 && strcmp(visVars{1},'y')
                    Y = visVals{1}; X = [];
                elseif numel(visVars == 2) && ismember('y',visVars) && ismember('x',visVars)
                    if strcmp(visVars{1},'y'), Y = visVals{1}; X = visVals{2};
                    else                       Y = visVals{2}; X = visVals{1}; end
                else  error('You can only condition on Y and optionally X'); 
                end
            else
                if ~ischar(visVars) || ~strcmpi(visVars,'y')
                   error('You must condition on the observation sequence ''Y'''); 
                end
                Y = visVals;  X = [];
            end
            Z = MvnDist(); ZZ = MvnDist();
            [Z.mu, Z.Sigma, ZZ.Sigma, loglik] = model.infMethod(Y       ,...
                model.sysMatrix       , model.obsMatrix                 ,...
                cov(model.sysNoise)   , cov(model.obsNoise)             ,...
                mean(model.startDist) , cov(model.startDist)            ,...
                'u',X,'B',model.inputMatrix,'model',model.modelSwitch  );
            ZZ.mu = Z.mu;
        end
            
        function [Z,Y] = sample(model,nTimeSteps,controlSignal)
        % Simulate a run of a (switching) stochastic linear dynamical
        % system. Z are the hidden states and Y are the observations. You
        % can optionally specify a controlSignal used in conjuction with
        % model.inputMatrix
        %
        % Z(t+1) = sysMatrix*Z(t) + inputMatrix*controlSignal(t) + w(t),  w ~ sysNoise,  z(0) ~ startDist
        %                       OR
        % Z(t+1) = sysMatrix*Z(t) + w(t),  w ~ sysNoise,  z(0) ~ startDist
        %
        % Y(t) =   obsMatrix*Z(t) + v(t),  v ~ obsNoise
        
            if nargin < 3, controlSignal = [];end
            [Z,Y] = sample_lds(...
                model.sysMatrix       , model.obsMatrix     ,...
                cov(model.sysNoise)   , cov(model.obsNoise) ,...
                mean(model.startDist) , nTimeSteps          ,...
                model.modelSwitch     , model.inputMatrix   ,...
                controlSignal                               );
            
        end
    end
        
    methods(Access = 'protected')
        
        function [model,A, C, Q, R, initz, initV,diagQ,diagR,ARmode] = initParams(model)
        % Initialize parameters prior to running EM.   
            diagQ  =  false; diagR  =  false;
            if ndims(model.sysMatrix) > 2, error('Fitting of a switching LDS is not supported'); end
            if (isempty(model.obsMatrix) || isempty(model.sysMatrix)) && isempty(model.stateSize)
                error('Please specify the dimensionality of the latent state space'); 
            end
            if isempty(model.sysMatrix), A = randn(model.stateSize);
            else                         A = model.sysMatrix;   end
            if isempty(model.obsMatrix), C = randn(model.obsSize,model.stateSize);
            else                         C = model.obsMatrix;   end
            if isempty(model.sysNoise)
               Q = 0.1*eye(model.stateSize);
               model.sysNoise = MvnDist(zeros(model.stateSize,1),Q);
            else
               Q = cov(model.sysNoise);
               diagQ = ~isempty(model.sysNoise.covtype) && ~strcmpi(model.sysNoise.covtype,'full');
            end
            if isempty(model.obsNoise)
               R = eye(model.obsSize); 
               model.obsNoise = MvnDist(zeros(model.obsSize,1),R);
            else
               R = cov(model.obsNoise);
               diagQ = ~isempty(model.obsNoise.covtype) && ~strcmpi(model.obsNoise.covtype,'full');
            end
            if isempty(model.startDist)
               initz = 10*rand(model.stateSize,1);
               initV = 10*eye(model.stateSize);
               model.startDist = MvnDist(initz,initV);
            else
               initz = mean(model.startDist);
               initV = cov(model.startDist);
            end
            ARmode = isequal(model.obsMatrix,eye(size(model.obsMatrix))) && ~any(colvec(cov(model.obsNoise)));
        end
            
            
    end
   
end