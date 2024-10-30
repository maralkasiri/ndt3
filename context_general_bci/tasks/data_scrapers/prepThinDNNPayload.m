function prepThinDNNDPayload(data, root_path_out)
    if nargin < 2 || isempty(root_path_out)
        root_path_out = 'D:\Data\manual';
    end

    sample_file = data.Files(1);
    if iscell(sample_file)
        sample_file = sample_file{1};
    end

    % Parse out the subject, location, and session number.
    pattern = '.*[\\/](CRS\d\d)([A-Za-z]+)\.data\.(\d+)[\\/].*\.Set(\d+)';
    tokens = regexp(sample_file, pattern, 'tokens');
    if isempty(tokens)
        disp('Failed to parse file path.');
        return;
    end
    tokens = tokens{1};
    subject = tokens{1};
    location = tokens{2};
    session = str2double(tokens{3});
    setNum = str2double(tokens{4});

    % Construct the set name and output path
    set_name = string(subject) + string(location) + ...
               "_session_" + num2str(session) + ...
               "_set_" + num2str(setNum);
    out_path = fullfile(root_path_out, set_name + ".mat");

    % Process the data
    try
        thin_data = [];

        % Apply blacklist criteria
        if any(data.stim_idx)
            disp('Data excluded due to stim_idx criteria.');
            return;
        end

        % Process and cast data
        thin_data.SpikeCount = cast(data.SpikeCount, 'uint8');
        thin_data.SpikeCount = thin_data.SpikeCount(:,1:5:end);
        thin_data.trial_num = cast(data.trial_num, 'uint8');
        thin_data.passed = data.XM.passed;

        if isfield(data.Kinematics, 'ActualPos') || isfield(data.Kinematics, 'ActualForce')
            if isfield(data.Kinematics, 'ActualPos')
                thin_data.pos = cast(data.Kinematics.ActualPos(:,1:14), 'single'); % Can go up to 14 for finger
                if size(thin_data.pos, 1) ~= size(thin_data.SpikeCount, 1)
                    disp("mismatched shape, drop " + set_name);
                    clearvars thin_data.pos; % abandon attempt
                end
            end

            if isfield(data.Kinematics, 'ActualForce')
                thin_data.force = cast(data.Kinematics.ActualForce(:,1:1), 'single'); % Though this is 5D, I'm told by GB that essentially only dim 1 (index finger point force sensor in Mujoco) is rendered
            end
            % 1:3 - only take right hand's worth of control domains
            % (translation, rotation, grasp)
            thin_data.brain_control = cast(data.TaskStateMasks.brain_control_weight(1:3, :)', 'single');
            thin_data.brain_control(isnan(thin_data.brain_control)) = 0;
            thin_data.active_assist = cast(data.TaskStateMasks.active_assist_weight(1:3, :)', 'single');
            thin_data.active_assist(isnan(thin_data.active_assist)) = 0;
            thin_data.passive_assist = cast(data.TaskStateMasks.passive_assist_weight(1:3, :)', 'single');
            thin_data.passive_assist(isnan(thin_data.passive_assist)) = 0;

            if isfield(data.TaskStateMasks, 'active_override')
                if sum(data.TaskStateMasks.active_override(1:14, :), "all", "omitnan") > 0
                    thin_data.override = cast(data.TaskStateMasks.active_override(1:14, :)', 'single');
                    thin_data.override(isnan(thin_data.override)) = 0;
                end
            end
        end
        save(out_path, 'thin_data');
    catch ME
        disp(['Error processing and saving data: ', ME.message]);
    end

end