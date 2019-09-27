function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

    predictions = (pval < epsilon);

    tp = 0;
    fp = 0;
    fn = 0;

    for i = 1 : size(predictions)

        if predictions(i) == 1 && yval(i) == 1 
            tp += 1;
        end

        if predictions(i) == 1 && yval(i) == 0
            fp += 1;
        end

        if predictions(i) == 0 && yval(i) == 1
            fn += 1;
        end

    end


    prec = tp / (tp + fp);
    rec = tp / (tp + fn);

    F1 = 2 * prec * rec / (prec + rec);

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
