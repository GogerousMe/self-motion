function result = generateGratingSequence(outputDir,cfg)
% All speeds/directions/black segments in ONE DAT; image coordinates (90=down).
% No input DATs, external classes or toolboxes. Close window to skip preview.
validateattributes(cfg.Size,{'numeric'},{'vector','numel',2,'integer','>=',2,'finite'});
validateattributes(cfg.Directions,{'numeric'},{'vector','nonempty','real','finite','>=',0,'<',360});
validateattributes(cfg.Speeds,{'numeric'},{'vector','nonempty','real','finite','nonnegative'});
validateattributes(cfg.PeriodPx,{'numeric'},{'scalar','real','finite','>',2});
validateattributes(cfg.StimulusFrames,{'numeric'},{'scalar','integer','positive','finite'});
validateattributes(cfg.BlankFrames,{'numeric'},{'scalar','integer','nonnegative','finite'});
validateattributes(cfg.FrameMs,{'numeric'},{'scalar','positive','finite'});
validateattributes(cfg.PhaseCycles,{'numeric'},{'scalar','real','finite'});
validateattributes(cfg.PlaybackRate,{'numeric'},{'scalar','positive','finite'});
validateattributes(cfg.ShowPreview,{'logical'},{'scalar'});
validateattributes(cfg.TrailingBlank,{'logical'},{'scalar'});
assert(all(cfg.Speeds/cfg.PeriodPx<0.5),'Speed must be below half a period/frame.');
assert(numel(unique(cfg.Speeds))==numel(cfg.Speeds),'Speeds must be unique.');
assert(~isfolder(outputDir) && ~isfile(outputDir),'Choose a new output directory.');
[ok,msg]=mkdir(outputDir); assert(ok,'%s',msg);
height=cfg.Size(1); width=cfg.Size(2);
nDir=numel(cfg.Directions);
nConditions=nDir*numel(cfg.Speeds);
total=nConditions*cfg.StimulusFrames+(nConditions-double(~cfg.TrailingBlank))*cfg.BlankFrames;
[x,y]=meshgrid(0:width-1,0:height-1);
blank=zeros(height,width,'uint8');
preview=cfg.ShowPreview; fig=[]; img=[]; caption=[];
if preview
    fig=figure('Name','Grating preview - close to skip preview','NumberTitle','off');
    img=imagesc(blank,[0 255]); colormap(fig,gray(256)); axis image off;
    caption=title('Preparing');
end
previewCleanup=onCleanup(@() closePreview(fig)); %#ok<NASGU>
name='all_stimuli';
path=fullfile(outputDir,[name '.dat']);
frames=zeros(height,width,total,'uint8');
rows=zeros(nConditions*2,8); row=0; cursor=0;
for s=1:numel(cfg.Speeds)
    speed=cfg.Speeds(s);
    for a=1:nDir
        angle=cfg.Directions(a);
        projection=x*cosd(angle)+y*sind(angle);
        row=row+1;
        rows(row,:)=[row angle speed 0 cursor+1 cursor+cfg.StimulusFrames ...
            cursor*cfg.FrameMs (cursor+cfg.StimulusFrames)*cfg.FrameMs];
        for t=0:cfg.StimulusFrames-1
            halfCycle=2*((projection-speed*t)/cfg.PeriodPx+cfg.PhaseCycles);
            nearest=round(halfCycle);
            near=abs(halfCycle-nearest)<1e-10;
            halfCycle(near)=nearest(near);
            frame=uint8(mod(floor(halfCycle),2)==0)*255;
            cursor=cursor+1; frames(:,:,cursor)=frame;
            showFrame(frame,angle,false,t+1,cfg.StimulusFrames);
        end
        if cfg.BlankFrames>0 && ((s-1)*nDir+a<nConditions || cfg.TrailingBlank)
            row=row+1;
            rows(row,:)=[row angle 0 1 cursor+1 cursor+cfg.BlankFrames ...
                cursor*cfg.FrameMs (cursor+cfg.BlankFrames)*cfg.FrameMs];
            for b=1:cfg.BlankFrames
                cursor=cursor+1;
                showFrame(blank,angle,true,b,cfg.BlankFrames);
            end
        end
    end
end
assert(cursor==total);
writeAndCheckDat(path,frames);
segments=array2table(rows(1:row,:),'VariableNames', ...
    {'segment_index','direction_slot_deg','speed_px_per_frame','is_blank', ...
    'first_frame','last_frame','start_ms','end_ms'});
csvName=[name '_segments.csv']; writetable(segments,fullfile(outputDir,csvName));
runs=struct('name',name,'stimulus',[name '.dat'],'stimulus_sha256',sha256(path), ...
    'segments',csvName,'speeds_px_per_frame',cfg.Speeds, ...
    'frames',total,'duration_ms',total*cfg.FrameMs,'spikes',NaN);
fprintf('Checked ONE DAT %s: %d frames, %.2f s\n',name,total,total*cfg.FrameMs/1000);
result=struct('schema','matlab-sequential-binary-gratings-v2','config',cfg, ...
    'width',width,'height',height,'channels',1,'pixel_values',[0 255],'frame_ms',cfg.FrameMs, ...
    'coordinates','x-right_y-down; clockwise from right', ...
    'phase_policy','Reset phase at every direction; local t=0..StimulusFrames-1', ...
    'segment_policy','Frame indices 1-based inclusive; time intervals [start_ms,end_ms)', ...
    'condition_order','Speeds in configured order; within each speed, Directions in configured order; all in one DAT.', ...
    'direction_slot_note','Black rows have no motion; angle identifies preceding slot. Speed 0 has no motion direction.', ...
    'collection_note','Run the entire DAT continuously, including speed changes and black gaps. Black gaps do not reset V1.', ...
    'runs',runs);
fid=fopen(fullfile(outputDir,'manifest.json'),'w','n','UTF-8'); assert(fid~=-1);
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(result,'PrettyPrint',true));

    function showFrame(frame,angle,isBlank,localFrame,count)
        if ~preview, return; end
        if ~isgraphics(fig), preview=false; return; end
        tick=tic; img.CData=frame;
        if isBlank, state='BLACK GAP'; else, state='GRATING'; end
        caption.String=sprintf('%s | %.3g px/frame | %g deg\nframe %d/%d | segment %d/%d | t %.2f s', ...
            state,speed,angle,cursor,total,localFrame,count,(cursor-1)*cfg.FrameMs/1000);
        drawnow;
        pause(max(0,cfg.FrameMs/1000/cfg.PlaybackRate-toc(tick)));
    end
end

function writeAndCheckDat(path,frames)
[height,width,count]=size(frames);
fid=fopen(path,'w','ieee-le'); assert(fid~=-1);
cleanup=onCleanup(@() fclose(fid));
assert(fwrite(fid,293390619,'int32')==1);
assert(fwrite(fid,1,'single')==1);
assert(fwrite(fid,0,'int32')==1); % GRATING_STIM
assert(fwrite(fid,1,'uint8')==1);
assert(fwrite(fid,[width height count],'int32')==3);
assert(fwrite(fid,permute(frames,[2 1 3]),'uint8')==numel(frames));
clear cleanup;
fid=fopen(path,'r','ieee-le'); assert(fid~=-1);
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
header=[fread(fid,1,'int32') fread(fid,1,'single') fread(fid,1,'int32') ...
    fread(fid,1,'uint8') fread(fid,3,'int32')'];
assert(isequal(header,[293390619 1 0 1 width height count]));
pixels=fread(fid,Inf,'*uint8'); expected=permute(frames,[2 1 3]);
assert(isequal(pixels,expected(:)),'DAT round-trip mismatch.');
assert(all(pixels==0 | pixels==255));
end

function hex=sha256(path)
fid=fopen(path,'r'); assert(fid~=-1);
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
bytes=fread(fid,Inf,'*uint8');
digest=java.security.MessageDigest.getInstance('SHA-256');
digest.update(typecast(bytes(:),'int8'));
hash=typecast(digest.digest(),'uint8');
hex=lower(reshape(dec2hex(hash,2).',1,[]));
end

function closePreview(fig)
if ~isempty(fig) && isgraphics(fig), close(fig); end
end
