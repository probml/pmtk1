function [W,fEvals] = L1ProjectionOrder1EML(X,Y,lambda,offsetAdded)
    %#eml
    % This is basically the same as L1GeneralProjection.m except it has been
    % significantly modified to make it emlmex compliant. Many of the options
    % available in L1GeneralProjection have been removed; this version hard codes
    % the gradient function and uses the order 1 method. The original code is by
    % Mark Schmidt. The changes were made by Matt Dunham.

    % Use in conjunction with runCompileAndSave().

    eml.extrinsic('matrixDivideEML');
    maxIter = 250;
    optTol = 1e-6;
    threshold = 1e-4;
    corrections = 100;

    w = zeros(2*size(X,2)*(size(Y,2)-1),1);
    p = length(w);
    nlambda = length(lambda);
    wP = w(1:nlambda);
    wM = w(nlambda+1:end);
    ww = wP-wM;
    [f,GRAD] = multinomLogregNLLGradHessL2(ww,X,Y,0,offsetAdded);
    f = f + sum(lambda.*wP) + sum(lambda.*wM);
    g = [GRAD;-GRAD] + [lambda.*ones(nlambda,1);lambda.*ones(nlambda,1)];
    g_prev = zeros(size(g));
    w_prev = zeros(size(w));
    B = eye(p);
    fEvals = 1;


    i = 1;
    while fEvals < maxIter

        f_old = f;
        w = w.*(abs(w) >= threshold);
        free = ~(w == 0 & g >=0);

        gsum = sum(abs(g).*free);

        if(gsum < optTol),break;end
        if(sum(free) == 0),break;end



        d = zeros(p,1);

        if i == 1
            %d(free==1) = -g(free==1);
            d = (d.*free) - g.*free;
            old_dirs = zeros(p,0);
            old_stps = zeros(p,0);
            Hdiag = 1;
        else
            y = g-g_prev;
            s = w-w_prev;

            numCorrections = size(old_dirs,2);
            if numCorrections < corrections
                % Full Update
                old_dirs(:,numCorrections+1) = s;
                old_stps(:,numCorrections+1) = y;
            else
                % Limited-Memory Update
                old_dirs = [old_dirs(:,2:corrections) s];
                old_stps = [old_stps(:,2:corrections) y];
            end

            % Update scale of initial Hessian approximation
            Hdiag = (y'*s)/(y'*y);

            % Find updates where curvature condition was satisfied
            %curvSat = sum(old_dirs(free==1,:).*old_stps(free==1,:)) > 1e-10;

            curvSat = sum(bsxfun(@times,old_dirs,free).*bsxfun(@times,old_stps,free)) > 1e-10;



            % Compute descent direction
            %d(free==1) = lbfgsC(-g(free==1),old_dirs(free==1,curvSat),old_stps(free==1,curvSat),Hdiag);

            d(free==1) = lbfgsC(-g.*free,old_dirs(free==1,curvSat),old_stps(free==1,curvSat),Hdiag);

        end




        %         if i == 1
        %             B = eye(p);
        %         else
        %             y = g-g_prev;
        %             s = w-w_prev;
        %             ys = y'*s;
        %
        %             if i == 2
        %                 if ys > 1e-10
        %                     B = ((y'*y)/(y'*s))*eye(p);
        %                 end
        %             end
        %             if ys > 1e-10
        %                 B = B + (y*y')/(y'*s) - (B*s*s'*B)/(s'*B*s);
        %             end
        %         end
        %
        %
        %
        %         %d(free==1) = -B(free==1,free==1)\g(free==1);
        %         d = matrixDivideEML(B,g,free);
        %
        %         g_prev = g;
        %         w_prev = w;


        gtd = g'*d;
        if gtd > -optTol,  break;   end


        % Try Newton step
        t = 1;
        % Adjust if step is too large
        if sum(abs(d)) > 1e5, t = 1e5/sum(abs(d));   end
        % Adjust on first iteration
        if  i == 1
            t = min(1,1/gsum);
        end

        [t,w,f,g,LSfunEvals] = ArmijoBacktrackEML(w,t,d,f,f,g,gtd,1e-4,2,optTol,lambda,X,Y,offsetAdded);
        fEvals = fEvals + LSfunEvals;

        % Project Results into non-negative orthant
        w = w.*(w >= 0);

        % Check Convergence Criteria
        if sum(abs(t*d)) < optTol, break; end

        if abs(f-f_old) < optTol, break; end

        if fEvals > maxIter, break; end
        i = i +1;
    end

    W = w(1:length(w)/2)-w(length(w)/2 + 1:end);

end

