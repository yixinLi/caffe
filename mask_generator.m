function mask_generator

mask_size = 24;

addpath /Users/yixinli/Documents/VOCdevkit/VOCcode/
VOCinit;

imgset= 'train';
ids=textread(sprintf(VOCopts.imgsetpath,imgset),'%s');

writepath =[VOCopts.datadir VOCopts.dataset '/Masks/'];
cd(writepath);

%% loop for every image
num_ids = size(ids);

for n = 1:num_ids(1)  
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{n}));
    img_height = rec.size.height;
    img_width = rec.size.width;
    num_objects = length(rec.objects);
    mask = zeros(mask_size, mask_size);
    
    for i = 1:mask_size
        for j = 1:mask_size
            T = [img_height/mask_size * (i-1),...
                 img_width/mask_size * (j-1),...
                 img_height/mask_size * i,...
                 img_width/mask_size * j];
            inter_prev = 0;
            for k = 1:num_objects
                inter = intersection(rec.objects(k).bbox, T);
                if inter > inter_prev
                    mask(i,j) = inter / area(T);
                end
                inter_prev = inter;
            end
        end
    end

    output_name = strcat(ids{n}, '_mask', '.tif');
    imwrite(mask, output_name); %% data type of mask: double, range: [0,1]
    
end



function inter = intersection(box, T)
if (box(1)<T(3) && box(2)<T(4)) && (T(1)<box(3) && T(2)<box(4))
    S1 = (box(1) - T(3)) * (box(2) - T(4));
    S2 = (T(1) - box(3)) * (box(2) - T(4));
	S3 = (T(1) - box(3)) * (T(2) - box(4));
    S4 = (box(1) - T(3)) * (T(2) - box(4));
	inter = min([S1, S2, S3, S4]);
else
	inter = 0;
end

function area = area(rectangle)
area = (rectangle(2) - rectangle(4)) * (rectangle(1) - rectangle(3));