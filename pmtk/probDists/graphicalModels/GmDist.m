classdef GmDist < ParamDist
  % graphical model
  
  properties
    G; %  a graph object
    domain;
    infMethod;
    infArgs = {};
  end

  %%  Main methods
  methods
     
    function d = ndimensions(obj)
       d = nnodes(obj.G); % size(obj.G,1);
    end
    
    function d = nnodes(obj)
       d = nnodes(obj.G); % size(obj.G,1);
    end
    
    function h = drawGraph(obj)
        h = draw(obj.G);
    end
      
    function postQuery = marginal(model, queryVars, varargin)
      % p(Q)
      postQuery = conditional(model, [], [], queryVars, varargin{:});
    end
      
    function [postQuery,logZ] = conditional(model, visVars, visVals, queryVars, varargin)
      % p(Q|V=v) where Q defaults to all the hidden variables
      domain = model.domain;
      if nargin < 4, queryVars = mysetdiff(domain, visVars); end
      [infMethod, infArgs] = process_options(varargin, ...
        'infMethod', model.infMethod, 'infArgs', model.infArgs);
      switch lower(infMethod)
        case 'varelim'
          if ~iscell(queryVars)
            [postQuery, logZ] = varElimInf(model, visVars, visVals, queryVars, infArgs{:});
          else
            for q=1:length(queryVars)
              [postQuery{q}, logZ(q)] = varElimInf(model, visVars, visVals, ...
                queryVars{q}, infArgs{:});
            end
          end
        case 'enum'
          % only works if everything is discrete
          % Compute joint once. Then marginalize for each query.
          Tfac = convertToJointTabularFactor(model);
          [Tfac, Z] = normalizeFactor(slice(Tfac, visVars, visVals)); % p(H,v)
          logZ = log(Z);
          if ~iscell(queryVars)
             postQuery = marginalize(Tfac, queryVars);
          else
            for q=1:length(queryVars)
              postQuery{q} = marginalize(Tfac, queryVars{q});
              logZ(q) = log(Z);
            end
          end
        case 'gibbs'
           % Compute samples once. Then marginalize for each query.
          samples = gibbsInf(model, visVars, visVals);
          logZ = []; % hard to compute Z with MCMC...
          if ~iscell(queryVars)
            postQuery = marginal(samples, queryVars);
          else
            for q=1:length(queryVars)
              postQuery{q} = marginal(samples, queryVars{q});
            end
          end
        otherwise
          error(['unrecognized method ' model.infMethod])
      end
    end
    
  end
  
   methods(Access = 'protected')
    function [postQuery, logZ] = varElim(model, visVars, visVals, queryVars)
      % Here we have hard-coded the assumption variables are discrete!
      [Tfac,nstates] = convertToTabularFactors(model,visVars,visVals);
      cfacs = (cellfun(@(x)isequal(pmf(x),1),Tfac));
      % factors involving continuous, unobserved nodes
      if any(cfacs)
        %postQuery = varElimCts(model, Tfac, cfacs, queryVars)
        error('cannot handle cts latent nodes (even children)')
      end
      if(model.G.directed)
        moralGraph = moralize(model.G);
      else
        moralGraph = model.G;
      end
      ordering = best_first_elim_order(moralGraph.adjMat,nstates);
      elim = mysetdiff(mysetdiff(model.domain(ordering),queryVars),visVars);
      [postQuery, Z] = normalizeFactor(variableElimination(Tfac, elim));
      logZ = log(Z);
    end
    
    
    function [samples, convDiag] = gibbsInf(model, visVars, visVals, ...
        Nsamples, Nchains, Nburnin, thin, verbose)
      fullCond = makeFullConditionals(model, visVars, visVals);
      xinit = mcmcInitSample(model, visVars, visVals);
      ndims = length(xinit);
      samples = zeros(Nsamples, ndims, Nchains);
      for c=1:Nchains
        if verbose
          fprintf('starting to collect %d samples from chain %d of %d\n', ...
            Nsamples, c, Nchains);
        end
        if c>1, xinit = mcmcInitSample(model, visVars, visVals); end
        [samples(:,:,c)] = gibbsSample(fullCond, xinit, Nsamples, Nburnin, thin);
      end
      if Nchains > 1
        [convDiag.Rhat, convDiag.converged] = epsrMultidim(samples);
        samples = permute(samples, [1 3 2]); % s,c,j
        samples = reshape(samples, Nsamples*Nchains, ndims); % s*c by j
      end
       % The samples only contain values of the hidden variables, not all
       % the variables, so we need to 'label' the columns with the right
       % domain, so subsequent calls to marginal will work correctly.
       hidVars = mysetdiff(model.domain, visVars);
       samples = SampleDist(samples, hidVars); % , model.support(hidVars));
      end
    
    
    
  end

end