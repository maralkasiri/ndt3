% Used to pull successful cursor runs
sets = [1, 16];
sets = [1, 9, 10, 21];
% sets = [11,12,17,18];
% sets = [12,13,14,15,41,45,48,49];
% sets = [15,16,18,21];
% session_dir = "P:\users\joy47\PTest.data.00063/";
session_dir = "P:\data_raw\human\crs_array\P4\MotorExperiments\P4Lab.data.00077";
session_dir = "P:\data_raw\human\crs_array\P4\MotorExperiments\P4Lab.data.00085";

sets = [2, 3, 6];
session_dir = 'P:\data_raw\human\crs_array\P4\MotorExperiments\P4Lab.data.00059';

sets = [1, 2];
session_dir = 'P:\data_raw\human\crs_array\P4\MotorExperiments\P4Lab.data.00056';
session_dir = 'P:\data_raw\human\crs_array\P4\MotorExperiments\P4Lab.data.00057';
sets = [1, 2];
session_dir = 'P:\data_raw\human\crs_array\P4\MotorExperiments\P4Lab.data.00058';

sets = [1, 2];
session_dir = 'P:\data_raw\human\crs_array\P2\MPL_Experiments\P2Lab.data.02048';

sets = [1, 2, 4, 7, 8];
session_dir = 'P:\data_raw\human\crs_array\P2\MPL_Experiments\P2Lab.data.02049';

sets = [11, 12, 13, 14, 15, 16, 17, 18, 19];
session_dir = 'P:\data_raw\human\crs_array\P2\MPL_Experiments\P2Lab.data.02191';

out_path = 'D:\Data\analysis';
mkdir(out_path);

% queries = ["\.Set0001\.*"];
for set_idx = 1:length(sets)
    set_num = sets(set_idx);
    pattern = sprintf('QL.*.Set%04d.*', set_num); % Constructs a regex pattern for the set, e.g., \.Set0022\.*
    full_pattern = fullfile(session_dir, pattern);
    [data] = prepData('files', full_pattern);

    prepThinDNNPayload(data, out_path, true);
end
