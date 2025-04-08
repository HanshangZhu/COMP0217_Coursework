% myEKF_nomag.m main ekf functions
function [X_Est, P_Est, GT , gyro_z] = myEKF_nomag(out, q, r,plt)
    %% Sensor Data Extraction
    accel = squeeze(permute(out.Sensor_ACCEL.signals.values, [3, 2, 1]));
    gyro  = squeeze(permute(out.Sensor_GYRO.signals.values, [3, 2, 1]));
    accel2 = squeeze(permute(out.Sensor_LP_ACCEL.signals.values, [3, 2, 1]));

    tof_front        = squeeze(out.Sensor_ToF2.signals.values(:,1,:));
    tof_front_status = squeeze(out.Sensor_ToF2.signals.values(:,4,:));
    tof_left         = squeeze(out.Sensor_ToF3.signals.values(:,1,:));
    tof_left_status  = squeeze(out.Sensor_ToF3.signals.values(:,4,:));
    tof_right        = squeeze(out.Sensor_ToF1.signals.values(:,1,:));
    tof_right_status = squeeze(out.Sensor_ToF1.signals.values(:,4,:));

    pos     = out.GT_position.signals.values;
    rotQuat = out.GT_rotation.signals.values;
    timeVec = squeeze(out.GT_position.time);

    sensor_calibration = load("sensor_calibration.mat");

    N = length(pos);
    GT = pos;

    accel = (sensor_calibration.Rotw_a * accel')';
    gyro  = (sensor_calibration.Rotw_a * gyro')';
    accel2 = (sensor_calibration.Rotw_a2 * accel2')';

    accel = accel - sensor_calibration.accel_bias;
    gyro  = gyro - sensor_calibration.gyro_bias;
    accel2 = accel2 - sensor_calibration.accel2_bias;

    fs1 = 104; fs2 = 100; fc = 0.5; order=2;
    [b1, a1] = butter(order, fc/(fs1/2));
    [b2, a2] = butter(order, fc/(fs2/2));
    accel1_x = filtfilt(b1, a1, accel(:,1));
    accel1_y = filtfilt(b1, a1, accel(:,2));
    accel2_x = filtfilt(b2, a2, accel2(:,1));
    accel2_y = filtfilt(b2, a2, accel2(:,2));

    w1 = 2; w2 = 1;
    fused_accel_x = (w1 * accel1_x + w2 * accel2_x) / (w1 + w2);
    fused_accel_y = (w1 * accel1_y + w2 * accel2_y) / (w1 + w2);
    accel_fused = [fused_accel_x, fused_accel_y, accel(:,3)];
    if plt
        figure()
        plot(fused_accel_y)
        hold on
        plot(fused_accel_x)
        hold off
        title('Smoothed fused Accel x/y');
        legend('y','x');
       
    end

    %% Low-pass filter gyro yaw rate
    order = 5; cutoff = 2;
    [b, a] = butter(order, cutoff / (104/2));
    gyro_z = filtfilt(b, a, gyro(:,3));
    %gyro(:,3) = gyro_z;  % Optional: replace raw gyro_z in original array if needed

    %% Debug: Plot filtered vs raw gyro yaw rate
    if plt
        figure;
        plot(timeVec, gyro(:,3), 'b-', 'DisplayName', 'Raw Gyro Z'); hold on;
        plot(timeVec, gyro_z, 'r-', 'DisplayName', 'Filtered Gyro Z');
        legend();
        xlabel('Time [s]');
        ylabel('Yaw Rate [rad/s]');
        title('Gyro Z-axis (Yaw Rate) - Raw vs Filtered');
        grid on;
    end


    yawGT = quat2eul(rotQuat);
    yawGT = yawGT(:,1) + pi;
    GT = [GT, yawGT];

    X_Est = zeros(N, 6);
    P_Est = cell(N,1);
    X_Est(1,:) = [pos(1,1), pos(1,2), 0, 0, yawGT(2), 0];
    P_Est{1} = diag([0.001, 0.001, 0.05, 0.05, 0.001, 0.05]);
    Q = q;

    RdiagBase = [ sensor_calibration.R_tof_left, ...
                  sensor_calibration.R_tof_middle, ...
                  sensor_calibration.R_tof_right ];
    scaleFactors = r;

    disp(Q)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% UPDATE LOOP
    for k = 1:N-1
        dt = timeVec(k+1) - timeVec(k);
        if dt <= 0, dt = 1e-2; end

        xk = X_Est(k,:)';
        [x_pred, F] = predictionStepSymbolic(xk, accel_fused(k,:)', gyro_z(k), dt);
        P_pred = F * P_Est{k} * F' + Q;

        [z_meas, H] = measurementModelOptim(x_pred, ...
                    tof_right(k), tof_front(k), tof_left(k));
        R_local = diag(min(min([interpretToFStatus(tof_right_status(k)), ...
                        interpretToFStatus(tof_front_status(k)), ...
                        interpretToFStatus(tof_left_status(k))] .* (scaleFactors .* eye(3)*1000),0.1*scaleFactors),10) ); %manually disable the interpret status

        [x_upd, P_upd, K] = updateStep(x_pred, P_pred, z_meas, H, R_local);
        X_Est(k+1,:) = x_upd';
        P_Est{k+1} = P_upd;

        if mod(k,50) == 0
            fprintf('Step %d: Printing Q and R\n', k);
            disp(Q);
            disp(R_local);
            fprintf('Printing K:');
            disp(K);
        end
    end

    err_x = X_Est(:,1) - GT(:,1);
    err_y = X_Est(:,2) - GT(:,2);
    err_yaw = wrapToPi(X_Est(:,5) - GT(:,4));
    if plt
        figure;
        subplot(3,1,1); plot(timeVec, err_x); title('Error in x');
        subplot(3,1,2); plot(timeVec, err_y); title('Error in y');
        subplot(3,1,3); plot(timeVec, smooth(err_yaw)); title('Error in yaw');
    end
end
function [x_pred, F] = predictionStepSymbolic(xk, accel, gyro, dt)
    x   = xk(1);
    y   = xk(2);
    vx  = xk(3);
    vy  = xk(4);
    psi = xk(5);
    dpsi= xk(6);
    
    % Accelerometer in body frame
    a_x = accel(1);
    a_y = accel(2);
    R_bw = [cos(psi), -sin(psi); sin(psi), cos(psi)];
    a_world = R_bw * [a_x; a_y];
    
    a_wx = a_world(1);
    a_wy = a_world(2);
    gyZ = gyro;

    x_pred = zeros(6,1);
    x_pred(1) = x + vx*dt + 0.5*a_wx*dt^2;
    x_pred(2) = y + vy*dt + 0.5*a_wy*dt^2;
    x_pred(3) = vx + a_wx*dt;
    x_pred(4) = vy + a_wy*dt;
    x_pred(5) = psi + dpsi*dt;
    x_pred(6) = gyZ;

    F = eye(6);
    F(1,3) = dt;
    F(2,4) = dt;
    F(5,6) = dt;

    % NEW: Derivatives due to yaw rotation
    F(1,5) = -0.5 * dt^2 * (a_x * sin(psi) + a_y * cos(psi));
    F(2,5) =  0.5 * dt^2 * (a_x * cos(psi) - a_y * sin(psi));
    F(3,5) = -dt * (a_x * sin(psi) + a_y * cos(psi));
    F(4,5) =  dt * (a_x * cos(psi) - a_y * sin(psi));
end


function [z_meas, H] = measurementModelOptim(x_pred, d1, d2, d3)
    psi = x_pred(5); x = x_pred(1); y = x_pred(2);
    offsets = [-pi/2, 0, pi/2];
    angles = psi + offsets;
    d_pred = zeros(3,1);
    H = zeros(3,6);
    for i = 1:3
        theta = angles(i);
        [t, ~, dt_dx, dt_dy, dt_dpsi] = rayBoxIntersection(x, y, theta);
        d_pred(i) = t;
        H(i,1) = dt_dx;
        H(i,2) = dt_dy;
        H(i,5) = dt_dpsi;
    end
    z_meas = [d1; d2; d3];
end

function [x_upd, P_upd, K] = updateStep(x_pred, P_pred, z_meas, H, R)
    x_est = x_pred(1); y_est = x_pred(2); psi_est = x_pred(5);
    angles = psi_est + [-pi/2, 0, pi/2];
    d_pred = arrayfun(@(a) rayBoxIntersection(x_est, y_est, a), angles);
    z_pred = d_pred(:);

    y_tilde = z_meas - z_pred;
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;
    x_upd = x_pred + K * y_tilde;
    x_upd(5) = wrapToPi(x_upd(5));
    P_upd = (eye(length(x_pred)) - K * H) * P_pred;
end

function [t, branch, dt_dx, dt_dy, dt_dpsi] = rayBoxIntersection(x, y, theta)
    x_left = -1.2; x_right = 1.2;
    y_bottom = -1.2; y_top = 1.2;
    cth = cos(theta); sth = sin(theta);

    x_wall = ternary(cth > 0, x_right, ternary(cth < 0, x_left, NaN));
    y_wall = ternary(sth > 0, y_top, ternary(sth < 0, y_bottom, NaN));

    t_v = ternary(~isnan(x_wall) && abs(cth) > 1e-6, (x_wall - x)/cth, inf);
    t_h = ternary(~isnan(y_wall) && abs(sth) > 1e-6, (y_wall - y)/sth, inf);

    if t_v <= 0, t_v = inf; end
    if t_h <= 0, t_h = inf; end

    if t_v < t_h
        t = t_v; branch = 1;
        dt_dx = -1/cth; dt_dy = 0;
        dt_dpsi = (x_wall - x) * sth / (cth^2);
    else
        t = t_h; branch = 2;
        dt_dx = 0; dt_dy = -1/sth;
        dt_dpsi = - (y_wall - y) * cth / (sth^2);
    end
end

function scale = interpretToFStatus(statusVal)
    switch statusVal
        case 0, scale = 1.0;
        case 2, scale = 2.0;
        case 4, scale = 4.0;
        otherwise, scale = 10.0;
    end
end

function val = ternary(cond, a, b)
    if cond
        val = a;
    else
        val = b;
    end
end
