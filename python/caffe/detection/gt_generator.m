% gtBoxes 20x1 cell, [instancex4]
% gtImIds 20x1 cell
% gtIms   20x1 cell
% testIms 4952x1 cell

%%
addpath('/mnt/neocortex/data/PASCAL/2007/VOCdevkit/VOCcode/')
addpath('/mnt/neocortex/scratch/bhwang/sharedhome/od2/DetectorB/')
annodir = '/mnt/neocortex/data/PASCAL/2007/VOCdevkit/VOC2007/Annotations/';
jpegdir = '/mnt/neocortex/data/PASCAL/2007/VOCdevkit/VOC2007/JPEGImages/';

wdir = '/mnt/neocortex/data/PASCAL/2007/VOCdevkit/';

cd(wdir)
VOCinit

%% VOC 2007
TotalIm = 9963;
rec = cell(TotalIm,1);

for i=1:TotalIm
    xmlfn = sprintf(VOCopts.annopath,sprintf('%06d',i));
    fprintf('%s\n',xmlfn);
    rec{i} = PASreadrecord(xmlfn);
end

%%
c = ceil(rand()*20);
[ids_tr,gt_tr] = textread(sprintf(VOCopts.clsimgsetpath,VOCopts.classes{c},'train'));
[ids_va,gt_va] = textread(sprintf(VOCopts.clsimgsetpath,VOCopts.classes{c},'val'));
[ids_te,gt_te] = textread(sprintf(VOCopts.clsimgsetpath,VOCopts.classes{c},'test'));

trIms = ids_tr;
vaIms = ids_va;
teIms = ids_te;

c
length(trIms)+length(vaIms)+length(teIms)
sum(gt_tr==0)
sum(gt_va==0)
sum(gt_te==0)

trainIms = cell(length(trIms),1);
for i=1:length(trIms)
    trainIms{i} = sprintf('%06d',trIms(i));
end
valIms = cell(length(vaIms),1);
for i=1:length(vaIms)
    valIms{i} = sprintf('%06d',vaIms(i));
end

%%
tr_gtImIds = cell(20,1);
tr_gtIms = cell(20,1);
va_gtImIds = cell(20,1);
va_gtIms = cell(20,1);

trIms = cat(2,trIms,(1:length(trIms))');
vaIms = cat(2,vaIms,(1:length(vaIms))');
for c=1:20
    [ids_tr,gt_tr] = textread(sprintf(VOCopts.clsimgsetpath,VOCopts.classes{c},'train'));
    [ids_va,gt_va] = textread(sprintf(VOCopts.clsimgsetpath,VOCopts.classes{c},'val'));
    
    % image referred to by their filename organized by their class
    % gtIms initialized here
    tr_gtIms_t{c} = ids_tr(gt_tr>0);
    for i=1:length(tr_gtIms_t{c})
        tr_gtIms{c}(i,1) = {sprintf('%06d',tr_gtIms_t{c}(i))};
    end
    va_gtIms_t{c} = ids_va(gt_va>0);
    for i=1:length(va_gtIms_t{c})
        va_gtIms{c}(i,1) = {sprintf('%06d',va_gtIms_t{c}(i))};
    end
    
    % gtImIds initialized here
    it = 1;
    for i=1:length(trIms)
        if tr_gtIms_t{c}(it,1) == trIms(i,1)
            tr_gtImIds{c}(it,1) = trIms(i,2);
            it = it+1;
            if it>length(tr_gtIms_t{c})
                break
            end
        end
    end
    it = 1;
    for i=1:length(vaIms)
        if va_gtIms_t{c}(it,1) == vaIms(i,1)
            va_gtImIds{c}(it,1) = vaIms(i,2);
            it = it+1;
            if it>length(va_gtIms_t{c})
                break
            end
        end
    end
end

%%
tr_it = 1;
va_it = 1;
te_it = 1;
trvateflags = [];
for i=1:TotalIm
    if tr_it <= size(trIms,1) && trIms(tr_it,1)==i
        trvateflags(i) = 0;
        tr_it = tr_it + 1;
    elseif va_it <= size(vaIms,1) && vaIms(va_it,1)==i
        trvateflags(i) = 1;
        va_it = va_it + 1;
    elseif te_it <= size(teIms,1) && teIms(te_it,1)==i
        trvateflags(i) = 2;
        te_it = te_it + 1;
    end
end

tr_gtBoxes = cell(20,1);
va_gtBoxes = cell(20,1);
te_gtBoxes = cell(20,1);
for i=1:TotalIm
    for j=1:length(rec{i}.objects)
        if rec{i}.objects(j).difficult == 1
            continue;
        end
        c = find(strcmp(VOCopts.classes, rec{i}.objects(j).class));
        
        switch trvateflags(i)
            case 0
                tr_gtBoxes{c} = cat(1,tr_gtBoxes{c}, rec{i}.objects(j).bbox);
            case 1
                va_gtBoxes{c} = cat(1,va_gtBoxes{c}, rec{i}.objects(j).bbox);
            case 2
                te_gtBoxes{c} = cat(1,te_gtBoxes{c}, rec{i}.objects(j).bbox);
            otherwise
                fprintf('trvateflags created wrong');
        end
    end
end
