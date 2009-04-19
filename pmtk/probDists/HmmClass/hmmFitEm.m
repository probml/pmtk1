
function model = hmmFitEm(model,data,varargin)
% Update all of the parameters of the HMM via EM
% We use the abriviation ess for expected sufficient statistics
[optTol,maxIter,clampedStart,clampedObs,clampedTrans] = ...
  process_options(varargin ,...
  'optTol'                ,1e-4   ,...
  'maxIter'               ,100    ,...
  'clampedStart'          ,false  ,...
  'clampedObs'            ,false  ,...
  'clampedTrans'          ,false  );
if(clampedStart && clampedObs && clampedTrans),return;end % nothing to do

[currentLL,prevLL,iter,converged,nobs,essStart,...
  essTrans,essObs,stackedData,seqndx,weightingMatrix] = initializeVariables();
% these variables are global to the emUpdate method



%% EM LOOP
while(iter <= maxIter && not(converged))
  resetStatistics();
  Estep();
  Mstep();
  testConvergence();
end

  function Estep()
    % Compute the expected sufficient statistics
    for j=1:nobs
      obs = getObservation(model,data,j);
      model = condition(model,'Y',obs);
      currentLL = currentLL + logprob(model);

      %% Starting Distribution
      if(not(clampedStart))
        [firstSlice,model] = marginal(model,1);          % marginal(model,1) is one slice marginal at t=1
        essStart.counts = essStart.counts + colvec(firstSlice);
      end
      %% Transition Distributions
      if(not(clampedTrans))
        [xi_summed,model] = marginal(model);                     % marginal(model) = full two slice marginals summed, i.e. xi_summed
        essTrans.counts = essTrans.counts + xi_summed;
      end
      if(not(clampedObs))
        [gamma,model] = marginal(model,':');                                        % marginal(model,':') all of the 1 slice marginals, i.e. gamma
        sz = size(gamma,2); idx = seqndx(j);
        weightingMatrix(idx:idx+sz-1,:) = weightingMatrix(idx:idx+sz-1,:) + gamma';  % seqndx just keeps track of where data cases start and finish  in the stacked matrix

      end
    end
    essTrans.counts = essTrans.counts';
    %% Emission Distributions

    if(not(clampedObs))
      for j=1:model.nstates
        essObs{j} = model.emissionDist{j}.mkSuffStat(stackedData,weightingMatrix(:,j));
      end
    end
  end % end of Estep subfunction

  function Mstep()
    % Maximize with respect to the ess calculated in the previous Estep
    %% Starting Distribution
    if(not(clampedStart))
      model.startDist = fit(model.startDist,'suffStat',essStart);
    end
    %% M Step Transition Matrix
    if(not(clampedTrans))
      model.transitionDist = fit(model.transitionDist,'suffStat',essTrans);
    end
    %% M Step Observation Model
    if(not(clampedObs))
      if(isTied(model.emissionDist{1})) % update the shared parameters first and then clamp them before updating the rest
        % since the state conditional densitity will know if
        % its tied or not, it can return appropriate suffStats.
        model.emissionDist{1} = fit(model.emissionDist{1},'suffStat',essObs{1});
        for i=2:model.nstates
          model.emissionDist{i} = unclampTied(fit(clampTied(model.emissionDist{i}),'suffStat',essObs{i}));
        end
      else
        for i=1:model.nstates
          model.emissionDist{i} = fit(model.emissionDist{i},'suffStat',essObs{i});
        end
      end
    end
  end % end of Mstep subfunction

  function testConvergence()
    % Test if EM has converged yet
    if(model.verbose && iter > 0)
      fprintf('\niteration %d, loglik = %f\n',iter,currentLL);
    end
    iter = iter + 1;
    converged = ((abs(currentLL - prevLL) / (abs(currentLL) + abs(prevLL) + eps)/2) < optTol) || (iter > maxIter);
  end % end of testConvergence subfunction

  function resetStatistics()
    % called during EM loop to reset stats
    prevLL = currentLL;
    currentLL = 0;
    if(~clampedStart)  ,essStart.counts(:) = 0;end
    if(~clampedTrans)  ,essTrans.counts(:) = 0;end
    if(~clampedObs)    ,weightingMatrix(:) = 0;end
  end % end of resetStatistics subfunction

  function [currentLL,prevLL,iter,converged,nobs,essStart,essTrans,essObs,stackedData,seqndx,weightingMatrix] = ...
      initializeVariables()
    % called prior to EM loop to setup variables
    currentLL = 0;
    prevLL = 0;
    iter = 0;
    converged = false;
    nobs  = nobservations(model,data);
    model = initializeParams(model,data);

    if(~clampedStart) ,essStart.counts = zeros(model.nstates,1)            ;end % The expected number of visits to state one - needed to update startDist
    if(~clampedTrans) ,essTrans.counts = zeros(model.nstates,model.nstates);end % The expected number of transitions from S(i) to S(j) - needed to update transDist
    if(~clampedObs)   , essObs = cell(model.nstates,1);end
    [stackedData,seqndx] = stackObservations(data);
    if(~clampedObs), weightingMatrix = zeros(size(stackedData,1),model.nstates);end
  end % end of initializeStatistics subfunction

end % end of emUpdate method



function model = initializeParams(model,X)
% Initialize parameters to starting states in preperation for EM.

if(model.initWithData)

  if(1) % initialize each component with all of the data, ignoring the temporal structure.
    data = stackObservations(X);
    if(allSameTypes(model.emissionDist))
      model.emissionDist = copy(fit(model.emissionDist{1},'data',data),1,model.nstates);
    else
      for i=1:model.nstates
        model.emissionDist{i} =  fit(model.emissionDist{i},'data',data);
      end
    end
  else % initialize each distribution with a random batch of data, ignoring temporal structure.
    nobs = nobservations(model,X);
    if(nobs >= model.nstates)
      for i=1:model.nstates
        model.emissionDist{i} =  fit(model.emissionDist{i},'data',getObservation(model,X,i)');
      end
    else
      data = stackObservations(X);
      n = size(data,1);
      batchSize = floor(n/model.nstates);
      if(batchSize < 2), batchSize = n;end
      for i=1:model.nstates
        model.emissionDist{i} =  fit(model.emissionDist{i},'data',data(sub(randperm(n),1:batchSize),:));
      end
    end
  end
end


end


function [X,ndx] = stackObservations(data)
% data is a cell array of sequences of different length but with the
% same dimensionality. X is a matrix of all of these sequences stacked
% together in an n-by-d matrix where n is the sum of the lengths of all
% of the sequences and d is the shared dimensionality. Within each cell
% of data, the first dimension is d and the second is the length of the
% observation. ndx stores the indices into X corresponding to the start
% of each new sequence.
%
% Alternatively, if data is a 3d matrix of size d-t-n, data is simply
% reshaped into size []-d and ndx is evenly spaced.

if(iscell(data))
  X = cell2mat(data)';
  ndx = cumsum([1,cell2mat(cellfuncell(@(seq)size(seq,2),data))]);
  ndx = ndx(1:end-1);
else
  X = reshape(data,[],size(data,1));
  ndx = cumsum([1,size(data,2)*ones(1,size(data,3))]);
  ndx = ndx(1:end-1);
end
end
end