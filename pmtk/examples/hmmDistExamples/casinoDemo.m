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
% We now display the rolls, the corresponding die used and the Viterbi 
% prediction. 
    dielabel = repmat('F',size(die));
    dielabel(die == 2) = 'L';
    vitlabel = repmat('F',size(viterbiPath));
    vitlabel(viterbiPath == 2) = 'L';
    rollLabel = num2str(rolls);
    rollLabel(rollLabel == ' ') = [];
    for i=1:60:300
        fprintf('Rolls:\t %s\n',rollLabel(i:i+59));
        fprintf('Die:\t %s\n',dielabel(i:i+59));
        fprintf('Viterbi: %s\n\n',vitlabel(i:i+59));
    end
%% Marginals 
% We can obtain filtered, (forwards only) and smoothed, (forwards/backwards) 
% estimates by calling marginal on the trellis. We specify ':' to return the
% entire path rather than values at specific points in the sequence.
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