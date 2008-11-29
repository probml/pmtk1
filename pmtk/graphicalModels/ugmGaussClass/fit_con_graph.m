function [Vg,l2,df]=fit_con_graph(V,num,cliques)
%GGM		Fit a graphical gaussian model.
%		V is the (co)variance matrix
%		num is the number of observations
%		cliques is a binary matrix specifying the model.
%		Example: [1 2] [2 3 4]
%		cliques=[1 1 0 0 
%		         0 1 1 1]
%		Vhat is the fitted covariance matrix.
%		l2 is the deviance with df degrees of freedom.

%#author Giovanni Marchetti
%#date 1994

tol=1e-9;
lold=+Inf;
converge=0;
[nc,k]=size(cliques);
Vg=eye(k); % Starting value
while ~converge
	for c=1:nc
		a=cliques(c,:);
		b=~a;
		Vaa=V(a,a);
%		Vbb=V(b,b);
%		Vba=V(b,a);
		Vgaa=Vg(a,a);
		Vgba=Vg(b,a);
		Vgbb=Vg(b,b);

		B=Vgba / Vgaa;
		Vpar=Vgbb - B* Vgaa * B';
% Update
		BV=B*Vaa;
		Vg(b,a)=BV;
		Vg(a,b)=BV'; 
		Vg(a,a)=Vaa;
		Vg(b,b)=Vpar + BV*B';
% Compute the deviance
		sd=V/Vg;
		l2=num*(trace(sd) - log(det(sd)) - k); 
		disp(l2)
% Check convergence
		if abs(lold-l2) < tol 
			converge=1;
			break
		else
			lold=l2;
		end
	end
end
% Degrees of freedom
df= sumall((~(cliques'*cliques)).*(~eye(k)))/2;





