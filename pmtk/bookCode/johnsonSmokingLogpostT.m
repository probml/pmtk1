function val = johnsonSmokingLogpostT(xy)
% Smoking/cancer example from Johnson and Albert p51
% val(i) = log posteror with xy(i,:) is i'th param vector (alpha, eta)

y1 = 83; n1 = 86; y2 = 72; n2 = 86;  %(data on p35)
alpha=xy(:,1);   eta=xy(:,2);
t1 = (alpha+eta)/2; t2 = (eta-alpha)/2;
val = y1*t1 - n1*log(1+exp(t1))  + y2*t2 - n2*log(1+exp(t2));
