function [c,ceq] = conQ2(x)
c(1) = x(1)^2 - x(2) + 1;
c(2) = 1 - x(1) + (x(2) -4)^2;
ceq = [];