% 直接点击 Run：所有速度、所有方向和黑场写入同一个 DAT，同时播放。
% 方向：0右、45右下、90下、135左下、180左、225左上、270上、315右上。
clear; clc;
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
cfg.Size = [80 80];
cfg.Directions = 0:45:315;
cfg.Speeds = [1.5 0.5 1 2];    % 像素/帧；只测试一种速度可写 1.5
% 添加速度0可将静态光栅段追加到同一个DAT。静态光栅不是纯黑帧。
cfg.PeriodPx = 12;             % 完整黑白周期，约6白+6黑
cfg.StimulusFrames = 100;      % 每个方向的光栅帧数
cfg.BlankFrames = 50;          % 每个方向之后的纯黑帧数
cfg.TrailingBlank = true;      % 最后一个方向之后也保留黑帧
cfg.FrameMs = 50;              % 与 CARLsim 帧时长一致
cfg.PhaseCycles = 0;           % 每个方向开始时重置相位
cfg.ShowPreview = true;        % false只生成、不播放
cfg.PlaybackRate = 1;          % 1原速，2倍速；不改变DAT或仿真时长
outputDir = fullfile(scriptDir,'bw_grating_output', ...
    ['sequence_' char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'))]);
result = generateGratingSequence(outputDir,cfg);
fprintf('输出目录：%s\n',outputDir);
