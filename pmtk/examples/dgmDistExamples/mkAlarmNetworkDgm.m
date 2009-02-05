%% Create the Alarm Network DGM
%#testPMTK
function dgm = mkAlarmNetworkDgm()
    
    load alarmNetwork
    N = numel(alarmNetwork.CPT);
    CPDs = cell(N,1);
    for i=1:N
        CPDs{i} = TabularCPD(alarmNetwork.CPT{i});
    end
    dgm = DgmDist(alarmNetwork.G, 'CPDs', CPDs);
    dgm.domain = 1:N;
end