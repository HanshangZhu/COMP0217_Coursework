

% Load Accel / Gyro Training data
AccelGyroCalib = load("calib2_straight.mat");

% Load Magnetometer data
MagCalib = load("calib1_rotate.mat");


% Extract accelerometer and gyro data
accel = squeeze(permute(AccelGyroCalib.out.Sensor_ACCEL.signals.values, [3, 2, 1]));
gyro  = squeeze(permute(AccelGyroCalib.out.Sensor_GYRO.signals.values, [3, 2, 1]));
t = AccelGyroCalib.out.Sensor_ACCEL.time;

% Calib1 (rotate) time
t_mag = MagCalib.out.Sensor_MAG.time;


% Extract second accelerometer data
accel2 = squeeze(permute(AccelGyroCalib.out.Sensor_LP_ACCEL.signals.values, [3, 2, 1]));

% Extract Magnetometer Data (rotation)
mag = squeeze(permute(MagCalib.out.Sensor_MAG.signals.values, [3, 2, 1]));


% Do same for still data
mag_still = squeeze(permute(AccelGyroCalib.out.Sensor_MAG.signals.values, [3, 2, 1]));



% Extract TOF data
tof_left = squeeze(AccelGyroCalib.out.Sensor_ToF1.signals.values(:,1,:));
tof_middle  = squeeze(AccelGyroCalib.out.Sensor_ToF2.signals.values(:,1,:));
tof_right = squeeze(AccelGyroCalib.out.Sensor_ToF3.signals.values(:,1,:));


% Define rotation matrices
Rwa = [0 1 0;
       0 0 1;
      -1 0 0]; % Accelerometer / Gyro

Rwa2 = [1 0 0;
        0 0 -1;
        0 1 0]; % Accelerometer 2

Rwmag = [0 1 0;
         0 0 1;
         1 0 0]; % Magnetometer

accel_corrected = (Rwa * accel')'; 
gyro_corrected = (Rwa * gyro')';

accel2_corrected = (Rwa2 * accel2')';

mag_corrected = (Rwmag * mag')';


% Filter data where t < 60
idx = t < 60;  % Logical index where time is less than 60
t = t(idx);
accel_corrected = accel_corrected(idx, :);
gyro_corrected = gyro_corrected(idx,:);

% Extract TOF data and apply time filter
tof_left = tof_left(idx);
tof_middle = tof_middle(idx);
tof_right = tof_right(idx);

accel2_corrected = accel2_corrected(idx, :);

% Find Accelerometer and Gyro Bias and std
accel_bias = mean(accel_corrected);
gyro_bias = mean(gyro_corrected);

accel2_bias = mean(accel2_corrected);

R_accel = cov(accel_corrected);
R_gyro = cov(gyro_corrected);

R_accel2 = cov(accel2_corrected);

% Compute variances (1x1 covariance matrices)
R_tof_left = var(tof_left);
R_tof_middle = var(tof_middle);
R_tof_right = var(tof_right);


% Perform magnetometer calibration
[A_mag, b_mag, ~] = magcal(mag_corrected);

% A_mag: Transformation matrix
% b_mag: Bias vector

mag_corrected = (mag_corrected - b_mag) * A_mag;


% Use Magnetometer from static data to find std / covariance
R_mag = cov(mag_still);




% Plot results
% figure(1)
% plot(t, accel_corrected)
% title('Accelerometer Information')
% xlabel('Time (s)')
% ylabel('Acceleration (m/s^2)')
% legend('X', 'Y', 'Z')
% grid on
% 
% figure(2)
% plot(t, gyro_corrected);
% title('Gyroscope Data');
% xlabel('Time (s)');
% ylabel('Angular Velocity (rad/s)');
% legend('X (Roll)', 'Y (Pitch)', 'Z (Yaw)');
% 
% figure(3)
% plot(t, accel2_corrected)
% title('Accelerometer 2 Information')
% xlabel('Time (s)')
% ylabel('Acceleration (m/s^2)')
% legend('X', 'Y', 'Z')
% grid on
% 
% figure;
% plot(t_mag, mag_corrected);
% title('Calibrated Magnetometer Data');
% xlabel('Time (s)');
% ylabel('Magnetic Field (µT)');
% legend('X', 'Y', 'Z');

% Print Accelerometer Covariance Matrix
disp('Accelerometer Covariance Matrix (R_accel):');
disp(R_accel);

% Print Gyroscope Covariance Matrix
disp('Gyroscope Covariance Matrix (R_gyro):');
disp(R_gyro);

% Print Second Accelerometer Covariance Matrix
disp('Second Accelerometer Covariance Matrix (R_accel2):');
disp(R_accel2);

% Print Magnetometer Covariance Matrix
disp('Magnetometer Covariance Matrix (R_mag):');
disp(R_mag);

% Display Tof Covariances
disp('ToF Sensor Covariances:');
disp(['Left ToF: ', num2str(R_tof_left)]);
disp(['Middle ToF: ', num2str(R_tof_middle)]);
disp(['Right ToF: ', num2str(R_tof_right)]);

% Print Accelerometer Bias
disp('Accelerometer Bias (accel_bias):');
disp(accel_bias);

% Print Gyroscope Bias
disp('Gyroscope Bias (gyro_bias):');
disp(gyro_bias);

% Print Second Accelerometer Bias
disp('Second Accelerometer Bias (accel2_bias):');
disp(accel2_bias);

% Print Magnetometer Bias (from magcal)
disp('Magnetometer Bias (b_mag):');
disp(b_mag);

% Save to .mat file
save('sensor_calibration.mat', 'R_accel', 'R_gyro', 'R_accel2', 'R_mag', ...
    'accel_bias', 'gyro_bias', 'accel2_bias', 'b_mag', ...
    'R_tof_left', 'R_tof_middle', 'R_tof_right');