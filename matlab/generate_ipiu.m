clear;close all;

lr_folder = 'C:/Users/BoyuZ/OneDrive - Swinburne University/SmartAI/SR/data/ipiu/lr/train';
hr_folder = 'C:/Users/BoyuZ/OneDrive - Swinburne University/SmartAI/SR/data/ipiu/hr/train';

savepath = 'ipiu.h5';
size_input = 41;
size_label = 41;
stride = 41;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);

count = 0;
margain = 0;

%% generate data
lr_filepaths = [];
lr_filepaths = [lr_filepaths; dir(fullfile(lr_folder, '*.bmp'))];

hr_filepaths = [];
hr_filepaths = [hr_filepaths; dir(fullfile(hr_folder, '*.bmp'))];

for i = 1 : length(lr_filepaths)
    lr_image = imread(fullfile(lr_folder,lr_filepaths(i).name)); 
    hr_image = imread(fullfile(hr_folder,hr_filepaths(i).name)); 
    
%     image = imrotate(image, 90 * (degree - 1));
    
%     image = imresize(image,downsizes(downsize),'bicubic');
    
    if size(lr_image,3)==3
        lr_image = rgb2ycbcr(lr_image);
        lr_image = im2double(lr_image(:, :, 1));
    end
    if size(hr_image,3)==3
        hr_image = rgb2ycbcr(hr_image);
        hr_image = im2double(hr_image(:, :, 1));
    end
    
    [hei,wid] = size(lr_image);
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1

            subim_input = lr_image(x : x+size_input-1, y : y+size_input-1);
            subim_label = hr_image(x : x+size_label-1, y : y+size_label-1);

            count=count+1;
            imshow(subim_input)
            imshow(subim_label)

            data(:, :, 1, count) = subim_input;
            label(:, :, 1, count) = subim_label;
        end
    end


end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);
