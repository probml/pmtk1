%% Create the Flu Network DGM
%#testPMTK
function dgm = mkFluDgm(showGraph)

    if nargin == 0, showGraph = false; end
    season       = 1;
    flu          = 2;
    hayfever     = 3;
    muscle_pain  = 4;
    congestion   = 5;
    
    names = {'season','flu','hayfever','muscle pain','congestion'};
    
    G = zeros(5);
    G(season,[flu,hayfever]) = 1;
    G(flu,[muscle_pain,congestion]) = 1;
    G(hayfever,congestion) = 1;
    if(showGraph)
        Graphlayout('adjMatrix',G,'nodeLabels',names,'currentLayout',TreeLayout());
    end
    winter = 1;  spring  = 2;
    summer = 3;  autumn  = 4;
    absent = 1;  present = 2;
    
    % Somewhat arbitrary handset values
    CPT = cell(1,5);
    
    CPT{season} = normalize(ones(1,4));    % p(winter) = p(spring) = p(summer) = p(autumn) = 0.25
   
    CPT{flu}(winter,[absent , present]) = [0.90,0.10]; % p(flu | winter)    
    CPT{flu}(spring,[absent , present]) = [0.95,0.05]; % p(flu | spring)
    CPT{flu}(summer,[absent , present]) = [0.98,0.02]; % p(flu | summer)
    CPT{flu}(autumn,[absent , present]) = [0.93,0.07]; % p(flu | autumn)
    
    CPT{hayfever}(winter , [absent,present]) = [0.99,0.01]; % p(hayfever | winter)
    CPT{hayfever}(spring , [absent,present]) = [0.70,0.30]; % p(hayfever | spring)
    CPT{hayfever}(summer , [absent,present]) = [0.80,0.10]; % p(hayfever | summer)
    CPT{hayfever}(autumn , [absent,present]) = [0.90,0.10]; % p(hayfever | autumn)
    
    CPT{muscle_pain}(absent  , [absent,present])  = [0.9,0.1];  % p(muscle_pain | flu absent)
    CPT{muscle_pain}(present , [absent,present])  = [0.7,0.3];  % p(muscle_pain | flu present)
    
    CPT{congestion}(absent  , absent  , [absent,present]) = [0.98,0.02];  % p(congestion | flu absent , hayfever absent)
    CPT{congestion}(absent  , present , [absent,present]) = [0.20,0.80];  % p(congestion | flu absent , hayfever present)
    CPT{congestion}(present , absent  , [absent,present]) = [0.15,0.85];  % p(congestion | flu present, hayfever absent)
    CPT{congestion}(present , present , [absent,present]) = [0.05,0.95];  % p(congestion | flu present, hayfever present)
    
    CPD = cell(1,5);
    for c=1:numel(CPT)
        CPD{c} = TabularCPD(CPT{c});
    end
    
    dgm = DgmDist(G,'CPDs',CPD,'domain',1:5);
end