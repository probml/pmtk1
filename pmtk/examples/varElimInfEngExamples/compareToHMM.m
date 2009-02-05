%#inprogress
Z1 = 1; Z2 = 2;
Z3 = 3; Z4 = 4;
Y1 = 5; Y2 = 6;
Y3 = 7; Y4 = 8;

graph = zeros(8);
graph(Z1,Z2) = 1;
graph(Z2,Z3) = 1;
graph(Z3,Z4) = 1;
graph(Z1,Y1) = 1;
graph(Z2,Y2) = 1;
graph(Z3,Y3) = 1;
graph(Z4,Y4) = 1;
Graphlayout('adjMatrix',graph,'nodeLabels',{'Z1','Z2','Z3','Z4','Y1','Y2','Y3','Y4'},'currentLayout',Treelayout());

transmat = [0.9,0.1;0.1,0.9];


CPD{Z1} = TabularFactor([0.5;0.5]);
CPD{Z2} = TabularFactor(transmat);
CPD{Z3} = TabularFactor(transmat);
CPD{Z4} = TabularFactor(transmat);
CPD{Y1} = TabularFactor();
CPD{Y2} = TabularFactor();
CPD{Y3} = TabularFactor();
CPD{Y4} = TabularFactor();