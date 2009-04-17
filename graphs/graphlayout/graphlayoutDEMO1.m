load smallExample 
nodeColors = {'g','b','r','c'}; % if too few specified, it will cycle through
Graphlayout('adjMatrix',adj,'nodeLabels',names,'currentLayout',Treelayout,'nodeColors',nodeColors);
