%% HMMs and the Occasionally Dishonest Casino
% This is an example from 
% 
% 'Biological Sequence Analysis: 
% Probabilistic Models Proteins and Nucleic Acids' by Durbin, Eddy, Krogh, &
% Mitchison, (1998).
%%
% Suppose a casino uses a fair die most of the time but occasionally switches to
% and from a loaded die according to Markovian dynamics. We observe the dice
% rolls but not the type of die. We can use a Hidden Markov Model to predict
% which die is being used at any given point in a sequence of rolls. In this
% example, we know both the transition and emission probabilities. 
%% Specifying the Model
% Since we are not learning the parameters, we must specify the observation
% model, the transition matrix, and the distribution over starting states.
fair = 1; loaded = 2;
%% Observation Model  
% We will use a discrete observation model, one DiscreteDist object per hidden
% state of which there are two. We store these state conditional densities in a
% cell array.
    setSeed(0);
    obsModel = {DiscreteDist([1/6 , 1/6 , 1/6 , 1/6 , 1/6 , 1/6  ]);...   % fair die
                DiscreteDist([1/10, 1/10, 1/10, 1/10, 1/10, 5/10 ])};     % loaded die
%% Transition Matrix
% 
    transmat = [0.95  , 0.05;
                0.10  , 0.90];
%% Distribution over Starting States      
    pi = [0.5,0.5];
%% Create the Model
    model = HmmDist('pi',pi,'transitionMatrix',transmat,'stateConditionalDensities',obsModel);
%% Sample
% We now sample a single sequence of 300 dice rolls    
    nsamples = 1; length = 300;
    [rolls,die] = sample(model,nsamples,length);
%% Prediction
% We can obtain a distribution over paths, a TrellisDist, by calling predict on
% the fitted model, passing in a sequence of observations. The mode of this
% distribution is the Viterbi path or most likely sequence of hidden states. 
    trellis = predict(model,rolls);
    viterbiPath = mode(trellis);
%%
% This is different than the sequence of most likely states, which we can obtain
% by calling marginal() on the trellis and taking the max. The ':' returns the
% entire path rather than values at specific points in the sequence. These are
% the smoothed estimates, we can also obtain the filtered estimates for comparison.
% The method also supports two slice marginals, hence the []. 
   maxmarg = maxidx(marginal(trellis,':'));
   maxmargF = maxidx(marginal(trellis,':',[],'filtered'));
%%
% We can also sample from the posterior and compare the mode of these
% samples to the predictions above. 
   postSamp = mode(sample(trellis,500),2)';
%%
% We now display the rolls, the corresponding die used and the Viterbi 
% prediction. 
    dielabel = repmat('F',size(die));
    dielabel(die == 2) = 'L';
    vitlabel = repmat('F',size(viterbiPath));
    vitlabel(viterbiPath == 2) = 'L';
    maxmarglabel = repmat('F',size(maxmarg));
    maxmarglabel(maxmarg == 2) = 'L';
    postsamplabel = repmat('F',size(postSamp));
    postsamplabel(postSamp == 2) = 'L';
    rollLabel = num2str(rolls);
    rollLabel(rollLabel == ' ') = [];
    for i=1:60:300
        fprintf('Rolls:\t  %s\n',rollLabel(i:i+59));
        fprintf('Die:\t  %s\n',dielabel(i:i+59));
        fprintf('Viterbi:  %s\n',vitlabel(i:i+59));
        fprintf('MaxMarg:  %s\n',maxmarglabel(i:i+59));
        fprintf('PostSamp: %s\n\n',postsamplabel(i:i+59));
    end
%% 
viterbiErr  =  sum(viterbiPath ~= die);
maxMargSErr =  sum(maxmarg ~= die);
maxMargFErr =  sum(maxmargF~=die);
postSampErr    =  sum(postSamp ~= die);
fprintf('\nNumber of Errors\n');
fprintf('Viterbi:\t\t\t\t%d/%d\n',viterbiErr,300);
fprintf('Max Marginal Smoothed:  %d/%d\n',maxMargSErr,300);
fprintf('Max Marginal Filtered:  %d/%d\n',maxMargFErr,300);
fprintf('Mode Posterior Samples: %d/%d\n',postSampErr,300);

%% Marginals 
    filtered = marginal(trellis,':',[],'filtered'); % filtered(i,t) = p(S(t)=i | y(1:t))
    smoothed = marginal(trellis,':',[],'smoothed'); % smoothed(i,t) = p(S(t)=i | y(1:T))
%% 
% Here we plot the probabilities and shade in grey the portions of the die
% sequence where a loaded die was actually used. 
    figure; hold on;
    area(die-1,'FaceColor',0.75*ones(1,3),'EdgeColor',ones(1,3));
    plot(filtered(fair,:),'LineWidth',2.5);
    xlabel('roll number');
    ylabel('p(fair)');
    set(gca,'YTick',0:0.5:1);
    title(sprintf('filtered\n(grey bars correspond to a loaded die)'));
    
    figure; hold on;
    area(die-1,'FaceColor',0.75*ones(1,3),'EdgeColor',ones(1,3));
    plot(smoothed(fair,:),'LineWidth',2.5);
    xlabel('roll number');
    ylabel('p(fair)');
    set(gca,'YTick',0:0.5:1);
    title(sprintf('smoothed\n(grey bars correspond to a loaded die)'));
%%