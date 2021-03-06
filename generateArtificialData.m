function [O, Nm] = generateArtificialData(T)
    Tr = [0.9, 0.1, 0, 0; 0.1, 0.8, 0.1, 0; 0, 0.2, 0.7, 0.1; 0, 0.1, 0.3, 0.6;];
    H = [0.4, 0.2, 0.2, 0.2;...
         0.2, 0.4, 0.2, 0.2;...
         0.2, 0.2, 0.4, 0.2;...
         0.2, 0.2, 0.2, 0.4];
    pi = ones(1,4)/4;
    s = zeros(1, T);
    O = zeros(1, T);
    Nm = 4;
    s(1) = mnrnd(1, pi)*((1:4)');
    O(1) = mnrnd(1, H(s(1), :))*((1:4)');
    for i = 2:T
        s(i) = mnrnd(1, Tr(s(i-1), :))*((1:4)');
        O(i) = mnrnd(1, H(s(i), :))*((1:4)');
    end
end