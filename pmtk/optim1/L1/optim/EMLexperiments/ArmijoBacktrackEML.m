function [t,x_new,f_new,g_new,funEvals] = ArmijoBacktrackEML(x,t,d,f,fr,g,gtd,c1,LS,tolX,lambda,X,Y,offsetAdded)
%#eml
% See ArmijoBacktrackEML - this is special purpose version used only with
% L1ProjectionOrder1EML.m and compiled via emlmex. This code is almost entirely
% Mark Schmidt's with minor changes to make eml compliant by Matt Dunham. 

   eml.extrinsic('polyinterp');

    if any(isnan(d))
        t = 0;
        x_new = x;
        f_new = f;
        g_new = g;
        funEvals = 0;
        return;
    end


   
    w = x+t*d;
    for i=1:numel(w)
        if(w(i) < 0),w(i) = 0;end
    end
    nlambda = length(lambda);
    wP = w(1:nlambda);
    wM = w(nlambda+1:end);
    ww = wP-wM;
    [F,GRAD] = multinomLogregNLLGradHessL2(ww,X,Y,0,offsetAdded);
    f_new = F + sum(lambda.*wP) + sum(lambda.*wM);
    g_new = [GRAD;-GRAD] + [lambda.*ones(nlambda,1);lambda.*ones(nlambda,1)];





    f_prev = f_new;
    t_prev = t;


    funEvals = 1;

    while f_new > fr + c1*t*gtd || ~isLegal(f_new)

        temp = t;
        if LS == 0 || ~isLegal(f_new)
            % Backtrack w/ fixed backtracking rate

            t = 0.5*t;
        elseif LS == 2 && isLegal(g_new)
            % Backtracking w/ cubic interpolation w/ derivative

            t = polyinterp([0 f gtd; t f_new g_new'*d],0);
        elseif funEvals < 2 || ~isLegal(f_prev)
            % Backtracking w/ quadratic interpolation (no derivative at new point)

            t = polyinterp([0 f gtd; t f_new sqrt(-1)],0);
        else%if LS == 1
            % Backtracking w/ cubic interpolation (no derivatives at new points)

            t = polyinterp([0 f gtd; t f_new sqrt(-1); t_prev f_prev sqrt(-1)],0);
        end

        % Adjust if change in t is too small/large

        if t < temp*1e-3

            t = temp*1e-3;
        elseif t > temp*0.6

            t = temp*0.6;
        end

        f_prev = f_new;
        t_prev = temp;


        w = x+t*d;
        for i=1:numel(w)
            if(w(i) < 0),w(i) = 0;end
        end
        nlambda = length(lambda);
        wP = w(1:nlambda);
        wM = w(nlambda+1:end);
        ww = wP-wM;
        [F,GRAD] = multinomLogregNLLGradHessL2(ww,X,Y,0,offsetAdded);
        f_new = F + sum(lambda.*wP) + sum(lambda.*wM);
        g_new = [GRAD;-GRAD] + [lambda.*ones(nlambda,1);lambda.*ones(nlambda,1)];





        funEvals = funEvals+1;

        % Check whether step size has become too small
        if sum(abs(t*d)) <= tolX

            t = 0;
            f_new = f;
            g_new = g;
            break;
        end
    end



    x_new = x + t*d;

end
