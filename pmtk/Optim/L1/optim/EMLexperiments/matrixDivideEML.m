function d = matrixDivideEML(B,g,free)
    
    
    d = zeros(numel(free),1);
    d(free) = -B(free,free)\g(free);
    
    

    
end