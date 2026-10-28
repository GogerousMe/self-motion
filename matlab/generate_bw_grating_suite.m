function result = generate_bw_grating_suite(outputDir, varargin)
%GENERATE_BW_GRATING_SUITE Create new binary gratings for CARLsim V1.
% Run from the project root:
%   addpath('matlab');
%   result = generate_bw_grating_suite('outputs/matlab_bw_grating_01');
% All angles use image coordinates: 0 right, 90 down, clockwise positive.
% Speed is normal to the bars, in pixels/frame (not pixels/second).
% Written for MATLAB R2023b; no VisualStimulus classes or toolboxes required.

parser = inputParser;
addRequired(parser, 'outputDir', @(x) ischar(x) || isstring(x));
addParameter(parser, 'Size', [80 80], @(x) isnumeric(x) && numel(x)==2 && ...
    all(isfinite(x)) && all(x>=2) && all(x==fix(x)));
addParameter(parser, 'Frames', 80, @positiveInteger);
addParameter(parser, 'FrameMs', 50, @positiveInteger);
addParameter(parser, 'WarmupFrames', 20, @(x) isnumeric(x) && isscalar(x) && ...
    isfinite(x) && x>=0 && x==fix(x));
addParameter(parser, 'PeriodPx', 12, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x>2);
addParameter(parser, 'Speeds', [1.5 0.5 1 2], @(x) isnumeric(x) && isvector(x) && ...
    ~isempty(x) && all(isfinite(x)) && all(x>0));
addParameter(parser, 'PhaseCycles', 0, @(x) isnumeric(x) && isscalar(x) && isfinite(x));
addParameter(parser, 'PreviewGif', true, @(x) islogical(x) && isscalar(x));
parse(parser, outputDir, varargin{:});
cfg = parser.Results;
assert(cfg.WarmupFrames < cfg.Frames, 'Warmup must leave evaluation frames.');
assert(all(cfg.Speeds/cfg.PeriodPx < 0.5), ...
    'Speed must be below half a spatial period per frame.');
outputDir = char(outputDir);
assert(~isempty(outputDir), 'Output path must not be empty.');
assert(~isfolder(outputDir) && ~isfile(outputDir), ...
    'Output already exists; choose a new directory to preserve previous data.');
[ok, msg] = mkdir(outputDir);
assert(ok, '%s', msg);

height = cfg.Size(1);
width = cfg.Size(2);
[x,y] = meshgrid(0:width-1, 0:height-1);
angles = 0:45:315;
speeds = unique(cfg.Speeds, 'stable');
allRuns = struct([]);
suites = struct([]);
% Each moving speed has its own calibration-compatible manifest.
% Static controls have no motion-direction target and use a separate schema.
for group = 1:numel(speeds)+1
    isStatic = group > numel(speeds);
    if isStatic
        speed = 0;
        suiteName = 'static';
    else
        speed = speeds(group);
        suiteName = sprintf('speed_%s', strrep(sprintf('%.15g',speed),'.','p'));
    end
    folder = fullfile(outputDir, suiteName);
    mkdir(folder);
    runs = struct([]);
    contact = zeros(height, width*numel(angles), 'uint8');
    for a = 1:numel(angles)
        angle = angles(a);
        name = sprintf('bw_grating_d%03d', angle);
        frames = zeros(height, width, cfg.Frames, 'uint8');
        projection = x*cosd(angle) + y*sind(angle);
        for t = 0:cfg.Frames-1
            % White occupies half of each spatial period; no interpolation.
            % Snap floating point values close to a black/white boundary so
            % trig roundoff cannot introduce isolated boundary flicker.
            cycle = (projection-speed*t)/cfg.PeriodPx + cfg.PhaseCycles;
            halfCycle = 2*cycle;
            nearest = round(halfCycle);
            near = abs(halfCycle-nearest) < 1e-10;
            halfCycle(near) = nearest(near);
            frames(:,:,t+1) = uint8(mod(floor(halfCycle),2)==0)*255;
        end
        path = fullfile(folder, [name '.dat']);
        writeDat(path, frames);
        verifyDat(path, frames);
        imwrite(frames(:,:,1), fullfile(folder, [name '.png']));
        contact(:, (a-1)*width+(1:width)) = frames(:,:,1);
        if cfg.PreviewGif && group==1
            gifPath = fullfile(folder, [name '.gif']);
            for t = 1:cfg.Frames
                indexed = uint8(frames(:,:,t)>0);
                if t==1
                    imwrite(indexed, [0 0 0; 1 1 1], gifPath, 'gif', ...
                        'LoopCount', Inf, 'DelayTime', cfg.FrameMs/1000);
                else
                    imwrite(indexed, [0 0 0; 1 1 1], gifPath, 'gif', ...
                        'WriteMode', 'append', 'DelayTime', cfg.FrameMs/1000);
                end
            end
        end
        run = struct('name', name, 'expected_direction_deg', angle, ...
            'stimulus', [name '.dat'], 'stimulus_sha256', sha256(path), ...
            'spikes', NaN, 'grating_normal_deg', angle, ...
            'bar_orientation_deg', mod(angle+90,180), ...
            'speed_px_per_frame', speed, 'speed_px_per_second', speed*1000/cfg.FrameMs);
        if isStatic
            run.expected_direction_deg = NaN; % JSON null: motion undefined
        end
        if isempty(runs)
            runs = run;
        else
            runs(a) = run; %#ok<AGROW>
        end
        run.stimulus = [suiteName '/' name '.dat'];
        run.name = [suiteName '_' name];
        if isempty(allRuns)
            allRuns = run;
        else
            allRuns(end+1) = run; %#ok<AGROW>
        end
    end
    imwrite(contact, fullfile(folder, 'directions_first_frame.png'));
    manifest = struct('schema', 'v1-direction-calibration-v1', ...
        'width', width, 'height', height, 'frames', cfg.Frames, ...
        'frame_ms', cfg.FrameMs, 'duration_ms', cfg.Frames*cfg.FrameMs, ...
        'warmup_ms', cfg.WarmupFrames*cfg.FrameMs, ...
        'speed_px_per_frame', speed, 'kind', 'binary_square_grating', ...
        'period_px', cfg.PeriodPx, 'phase_cycles', cfg.PhaseCycles, ...
        'pixel_values', [0 255], 'coordinates', 'x-right_y-down', ...
        'directions', 'degrees clockwise from right', 'runs', runs, ...
        'instructions', ['Start an independent reset V1 run per DAT. ' ...
        'Use frame_ms for every frame, including warmup. ' ...
        'DAT contains no timing. spikes is a future output, not this stimulus.']);
    if isStatic
        manifest.schema = 'v1-static-grating-controls-v1';
    end
    writeJson(fullfile(folder,'manifest.json'), manifest);
    suiteEntry = struct('name', suiteName, 'manifest', ...
        [suiteName '/manifest.json'], 'conditions', numel(runs));
    if isempty(suites)
        suites = suiteEntry;
    else
        suites(group) = suiteEntry; %#ok<AGROW>
    end
end
result = struct('schema', 'matlab-binary-grating-suite-v1', ...
    'generator', 'generate_bw_grating_suite.m', 'config', cfg, ...
    'suites', suites, 'runs', allRuns, 'dat_count', numel(allRuns), ...
    'static_control_note', ['Each static normal angle shares the first frame ' ...
    'of every moving speed at that angle. Opposite static directions are ' ...
    'not independent motion conditions. No random seed is used.']);
writeJson(fullfile(outputDir, 'manifest.json'), result);
fprintf('Created and round-trip checked %d new DAT files in %s\n', ...
    result.dat_count, outputDir);
end

function yes = positiveInteger(x)
yes = isnumeric(x) && isscalar(x) && isfinite(x) && x>0 && x==fix(x);
end

function writeDat(path, frames)
% Image coordinates already use y-down. Transpose for C row-major order;
% DO NOT flipud here. MATLAB's fwrite otherwise writes columns first.
[height,width,count] = size(frames);
fid = fopen(path, 'w', 'ieee-le');
assert(fid~=-1, 'Cannot create %s', path);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
assert(fwrite(fid,293390619,'int32')==1);
assert(fwrite(fid,1,'single')==1);
assert(fwrite(fid,0,'int32')==1); % VisualStimulus GRATING_STIM
assert(fwrite(fid,1,'uint8')==1);
assert(fwrite(fid,[width height count],'int32')==3);
assert(fwrite(fid,permute(frames,[2 1 3]),'uint8')==numel(frames));
end

function verifyDat(path, expected)
fid = fopen(path,'r','ieee-le');
assert(fid~=-1, 'Cannot verify %s', path);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
assert(fread(fid,1,'int32')==293390619);
assert(fread(fid,1,'single')==1);
assert(fread(fid,1,'int32')==0);
assert(fread(fid,1,'uint8')==1);
dims = fread(fid,3,'int32');
assert(isequal(dims', [size(expected,2),size(expected,1),size(expected,3)]));
pixels = fread(fid,Inf,'*uint8');
assert(numel(pixels)==numel(expected));
actual = permute(reshape(pixels,dims'),[2 1 3]);
assert(isequal(actual,expected), 'DAT round trip failed.');
assert(all(actual(:)==0 | actual(:)==255), 'Nonbinary pixel found.');
end

function hex = sha256(path)
fid = fopen(path,'r');
assert(fid~=-1);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
bytes = fread(fid,Inf,'*uint8');
digest = java.security.MessageDigest.getInstance('SHA-256');
digest.update(typecast(bytes(:),'int8'));
hash = typecast(digest.digest(),'uint8');
hex = lower(reshape(dec2hex(hash,2).',1,[]));
end

function writeJson(path, data)
fid = fopen(path,'w','n','UTF-8');
assert(fid~=-1, 'Cannot create %s', path);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(data,'PrettyPrint',true));
end
