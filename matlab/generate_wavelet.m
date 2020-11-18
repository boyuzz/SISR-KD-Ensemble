clear;
close all;
folder = '../../data/RSSCN7/';
list_file = [folder, 'id_files.txt'];
savepath = 'wmcnn.h5';
file_length = 2800;
order = randperm(file_length);
training = order(1:1960);
testing = order(1961:2800);

%% scale factors
scale = 2;

size_label = 96;
size_input = size_label/scale;
stride = 48;

%% downsizing
downsizes = [1];

data = zeros(size_input, size_input, 1, 1);
Cs = zeros(size_label, size_label, 1, 1);
Dhs = zeros(size_label, size_label, 1, 1);
Dvs = zeros(size_label, size_label, 1, 1);
Dds = zeros(size_label, size_label, 1, 1);

count = 0;
margain = 0;

%% generate data
% mkdir([folder, 'test'])
fidin = fopen(list_file);
im_id = 1;
psnr_sum = 0;
while ~feof(fidin)
    tline=fgetl(fidin);
    imgpath = tline(3:end);
    filepath = [folder, imgpath];
    ori_image = imread(fullfile(folder,filepath));
    
    if ismember(im_id, testing)
        cls = strsplit(imgpath, '/');
        mkdir([folder, 'test/', cls{2}]);
%         imwrite(ori_image, [folder, 'test/', imgpath]);
        bic = imresize(imresize(ori_image, 1/scale, 'bicubic'), scale, 'bicubic');
        psnr_sum = psnr_sum + psnr(bic, ori_image);
    else
        for flip = 1: 2
            for degree = 1 : 4
                for downsize = 1 : length(downsizes)
                    image = ori_image;
                    if flip == 1
                        image = flipdim(image ,1);
                    end

                    image = imrotate(image, 90 * (degree - 1));
                    image = imresize(image,downsizes(downsize),'bicubic');

                    if size(image,3)==3
                        image = rgb2ycbcr(image);
                        image = im2double(image(:, :, 1));
                        im_label = modcrop(image, scale);
                        [hei,wid, c] = size(im_label);

    %                     filepaths(i).name
                        for x = 1 + margain : stride : hei-size_label+1 - margain
                            for y = 1 + margain :stride : wid-size_label+1 - margain
                                subim_label = im_label(x : x+size_label-1, y : y+size_label-1, :);
                                subim_input = imresize(subim_label,1/scale,'bicubic');
    %                             figure;
    %                             imshow(subim_input);
    %                             figure;
    %                             imshow(subim_label);
                                [cA,cH,cV,cD]=dwt2(subim_label,'haar');
                                count=count+1;
                                data(:, :, :, count) = subim_input;
                                Cs(:, :, :, count) = cA;
                                Dhs(:, :, :, count) = cH;
                                Dvs(:, :, :, count) = cV;
                                Dds(:, :, :, count) = cD;
                            end
                        end
                    end
                end
            end
        end
    end
    im_id = im_id + 1;
end
disp(psnr_sum/)
fclose(fidin);

order = randperm(count);
data = data(:, :, 1, order);
Cs = Cs(:, :, 1, order); 
Dhs = Dhs(:, :, 1, order); 
Dvs = Dvs(:, :, 1, order); 
Dds = Dds(:, :, 1, order); 

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
%     batchno
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);
    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);