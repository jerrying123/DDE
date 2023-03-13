load ICs.mat

figure;
for i = 1:length(X)
    if boolArr(i) == false
        plot(X(i,1),X(i,2), '*r');
        hold on;
    else
        plot(X(i,1),X(i,2), '+g');
        hold on;
    end
end
