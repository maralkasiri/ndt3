function pull_manual(root_path_out)
    if nargin < 1 || isempty(root_path_out)
        root_path_out = 'D:\Data\manual';
    end

    try
        data = prepData(); % enable data prompt
    catch
        disp('Failed to load data set.');
        return;
    end

    prepThinDNNPayload(data, root_path_out);
end
