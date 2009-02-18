function L = last(A)
    if isempty(A)
        L = [];
    else
        L = A(end);
    end
end