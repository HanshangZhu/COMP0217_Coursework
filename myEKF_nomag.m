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

        figure()
        plot(accel2_y) 
        hold on 
        plot(accel2_x)
        hold off
        title('OG Accel2 x/y');
        legend('y','x')

       
    end

    %% Low-pass filter gyro yaw rate
    filtgyro=1;
    if filtgyro
        order = 5; cutoff = 0.5;
        [b, a] = butter(order, cutoff / (104/2));
        gyro_z = filtfilt(b, a, gyro(:,3));
    end

    gyro_z = -gyro_z
    %gyro_z = -gyro(:,3)

    %% Debug: Plot filtered vs raw gyro yaw rate
    if plt && filtgyro
        figure;
        plot(timeVec, -gyro(:,3), 'b-', 'DisplayName', 'Raw Gyro Z'); hold on;
        plot(timeVec, gyro_z, 'r-', 'DisplayName', 'Filtered Gyro Z');
        legend();
        xlabel('Time [s]');
        ylabel('Yaw Rate [rad/s]');
        title('Gyro Z-axis (Yaw Rate) - Raw vs Filtered');
        grid on;
    end


    GT_eul = quat2eul(rotQuat);
    %yawGT = yawGT(:,1) + pi;
    yawGT = GT_eul(:,1);
    GT = [GT, yawGT];

    X_Est = zeros(N, 5);
    P_Est = cell(N,1);
    X_Est(1,:) = [pos(1,1), pos(1,2), 0, 0, yawGT(2)]; % convert yaw at 0 rads to relative to y axis
    P_Est{1} = diag([0.001, 0.001, 0.05, 0.05, 0.001]);
    Q = q;

    RdiagBase = [ sensor_calibration.R_tof_left, ...
                  sensor_calibration.R_tof_middle, ...
                  sensor_calibration.R_tof_right ];
    scaleFactors = r;

    disp(Q)
    
%% UPDATE LOOP
for k = 1:N-1
    dt = timeVec(k+1) - timeVec(k);
    if dt <= 0, dt = 1e-2; end

    xk = X_Est(k,:)';
    [x_pred, F] = predictionStepSymbolic(xk, accel_fused(k,:)', gyro_z(k), dt);
    P_pred = F * P_Est{k} * F' + Q;

    % Compute predicted measurements and Jacobian with the new model
    [z_pred, H] = measurementModel(x_pred);

    % Define actual measurements from TOF sensors
    z_meas = [tof_right(k); tof_front(k); tof_left(k)];

    % Compute local measurement noise covariance (unchanged)
    R_local = diag(scaleFactors);

    % Update step with actual and predicted measurements
    [x_upd, P_upd, K] = updateStep(x_pred, P_pred, z_meas, z_pred, H, R_local);
    X_Est(k+1,:) = x_upd';
    P_Est{k+1} = P_upd;

    if mod(k,50) == 0
        fprintf('Step %d:', k);
        %disp(Q);
        %disp(R_local);
        fprintf('z_pred:')
        disp(z_pred)
        fprintf('z_meas:')
        disp(z_meas)
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
    x = xk(1); y = xk(2); vx = xk(3); vy = xk(4); psi = xk(5);
    a_x = accel(1); a_y = accel(2);
    % Adjusted rotation matrix
    R_bw = [-sin(psi), -cos(psi); cos(psi), -sin(psi)];
    a_world = R_bw * [a_x; a_y];
    a_wx = a_world(1); a_wy = a_world(2); gyZ = gyro;

    x_pred = zeros(5,1);
    x_pred(1) = x + vx*dt + 0.5*a_wx*dt^2;
    x_pred(2) = y + vy*dt + 0.5*a_wy*dt^2;
    x_pred(3) = vx + a_wx*dt;
    x_pred(4) = vy + a_wy*dt;
    x_pred(5) = psi + gyZ*dt;

    F = eye(5);
    F(1,3) = dt; F(2,4) = dt;
    % Update Jacobian terms
    F(1,5) = -0.5 * dt^2 * (a_x * cos(psi) - a_y * sin(psi));
    F(2,5) = -0.5 * dt^2 * (a_x * sin(psi) + a_y * cos(psi));
    F(3,5) = -dt * (a_x * cos(psi) - a_y * sin(psi));
    F(4,5) = -dt * (a_x * sin(psi) + a_y * cos(psi));
end

function [x_upd, P_upd, K] = updateStep(x_pred, P_pred, z_meas, z_pred, H, R)
    % Compute innovation
    y_tilde = z_meas - z_pred;

    % Innovation covariance
    S = H * P_pred * H' + R;

    % Kalman gain
    K = P_pred * H' / S;

    % Update state estimate
    x_upd = x_pred + K * y_tilde;
    x_upd(5) =(x_upd(5)); % Ensure yaw stays within [-pi, pi]

    % Update covariance
    P_upd = (eye(length(x_pred)) - K * H) * P_pred;
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

function [z_pred, H] = measurementModel(x)
    x_pos = x(1); y_pos = x(2); theta = x(5); epsilon = 1e-4;
    x_max = 1.2; x_min = -1.2; y_max = 1.2; y_min = -1.2;
    z_pred = zeros(3,1); H = zeros(3,5);

    % Right sensor
    theta_right = theta + pi/2;
    dx_right = cos(theta_right);
    safe_dx_right = sign(dx_right) * max(abs(dx_right), epsilon);
    x_wall_right = ternary(dx_right >= 0, x_max, x_min);
    t_right = (x_wall_right - x_pos) / safe_dx_right;
    z_pred(1) = t_right;
    H(1,1) = -1 / safe_dx_right;
    H(1,5) = (x_wall_right - x_pos) * sin(theta_right) / (safe_dx_right^2);

    % Front sensor
    theta_front = theta + pi;
    dy_front = sin(theta_front);
    safe_dy_front = sign(dy_front) * max(abs(dy_front), epsilon);
    y_wall_front = ternary(dy_front >= 0, y_max, y_min);
    t_front = (y_wall_front - y_pos) / safe_dy_front;
    z_pred(2) = t_front;
    H(2,2) = -1 / safe_dy_front;
    H(2,5) = (y_wall_front - y_pos) * (-cos(theta)) / (safe_dy_front^2);

    % Left sensor
    theta_left = theta + 3*pi/2;
    dx_left = cos(theta_left);
    safe_dx_left = sign(dx_left) * max(abs(dx_left), epsilon);
    x_wall_left = ternary(dx_left >= 0, x_max, x_min);
    t_left = (x_wall_left - x_pos) / safe_dx_left;
    z_pred(3) = t_left;
    H(3,1) = -1 / safe_dx_left;
    H(3,5) = (x_wall_left - x_pos) * sin(theta_left) / (safe_dx_left^2);
end