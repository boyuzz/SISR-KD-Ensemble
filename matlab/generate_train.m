clear;
close all;
folder = '../data/291/';

savepath = 'stacksr.h5';

%% scale factors
scale = 2;

size_label = 96;
size_input = size_label/scale;
stride = 48;
downscale = [0.5, 0.7, 1];

data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);

count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

length(filepaths)

for i = 1 : length(filepaths)
    disp(i);
    ori_image = imread(fullfile(folder,filepaths(i).name));
    for flip = 1: 2
        for degree = 1 : 4
            for s = 1 : length(downscale)
                
                image = imresize(ori_image, downscale());
                
                if flip == 1
                    image = flipdim(image ,1);
                end

                image = imrotate(image, 90 * (degree - 1));

                if size(image,3)==3
                    image = rgb2ycbcr(image);
                    image = im2double(image(:, :, 1));
                    im_label = modcrop(image, scale);
                    [hei,wid, c] = size(im_label);

                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            subim_label = im_label(x : x+size_label-1, y : y+size_label-1, :);
                            subim_input = imresize(subim_label,1/scale,'bicubic');
    %                             figure;
    %                             imshow(subim_input);
    %                             figure;
    %                             imshow(subim_label);
                            count=count+1;
                            data(:, :, :, count) = subim_input;
                            label(:, :, :, count) = subim_label;
                        end
                    end
                end
            end
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