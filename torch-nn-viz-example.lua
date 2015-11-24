require 'torch'
require 'image'
require 'gnuplot'
require 'nn'

-- Default folder names
local filterResponsesFolderName = 'filter-responses'
local imageFolderName = 'images'

-- Very Simple test for gnuplot
local function gnuplotTest()
    a = image.lena();
    gnuplot.figure(1);
    gnuplot.imagesc(a[1])
end

-------------------------------------------------------------------------------
-----------------------TEST NETWORK DEFINITION---------------------------------
-------------------------------------------------------------------------------
-- Constructs and returns an inceptionModule from the paper 
-- "Going Deeper with Convolutional Networks", with input/output channels defined
-- with the parameters as follows:
-- inputChannels: the number of input channels
-- outputChannels: the expected number of outputChannels 
--                  (this parameter is only used to check the other parameters)
-- reductions: a 4-element array which specifies the number of channels output
--                  from each 1x1 convolutional network 
--                  (which should be smaller than the inputChannels usually...)
-- expansions: a 2-element array which specifies the number of channels output
--                  from the 3x3 convolutional layer and 
--                  the 5x5 convolutional layer
-- ReLU activations are applied after each convolutional layer
-- This module might be extended to allow for arbitrary width
local function inceptionModule(inputChannels, outputChannels, reductions, expansions)

    computedOutputChannels = reductions[1] + expansions[1] + expansions[2] + reductions[4]
    if not (outputChannels == computedOutputChannels) then
        print("\n\nOUTPUT CHANNELS DO NOT MATCH COMPUTED OUTPUT CHANNELS")
        print('outputChannels: ', outputChannels)
        print('computedOutputChannels: ', computedOutputChannels)
        print("\n\n")
        return nil
    end

    -- Remember, if there is no stacked first dimension (which here is just a
    -- single entry in the first dimension) then this should be 1.
    -- But since I reshape and add the empty first dimension, 
    -- I can keep this as 2.
    local inception = nn.DepthConcat(2)

    local column1 = nn.Sequential()
    column1:add(nn.SpatialConvolution(inputChannels, reductions[1],
        1, 1,  -- Convolution kernel
        1, 1)) -- Stride
    column1:add(nn.ReLU(true))
    inception:add(column1)
    
    local column2 = nn.Sequential()
    column2:add(nn.SpatialConvolution(inputChannels, reductions[2],
        1, 1, 
        1, 1))
    column2:add(nn.ReLU(true))
    column2:add(nn.SpatialConvolution(reductions[2], expansions[1],
        3, 3,  -- Convolution kernel
        1, 1)) -- Stride
    column2:add(nn.ReLU(true))
    inception:add(column2)

    local column3 = nn.Sequential()
    column3:add(nn.SpatialConvolution(inputChannels, reductions[3],
        1, 1, 
        1, 1))
    column3:add(nn.ReLU(true))
    column3:add(nn.SpatialConvolution(reductions[3], expansions[2],
        5, 5,  -- Convolution kernel
        1, 1)) -- Stride
    column3:add(nn.ReLU(true))
    inception:add(column3)

    local column4 = nn.Sequential()
    column4:add(nn.SpatialMaxPooling(3, 3, 1, 1))
    column4:add(nn.SpatialConvolution(inputChannels, reductions[4],
        1, 1,  -- Convolution kernel
        1, 1)) -- Stride
    column4:add(nn.ReLU(true))
    inception:add(column4)

    return inception
end

local function secondArch()

    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
        5, 5,
        1, 1))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialConvolution(64, 128, 
        3, 3,
        2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    -- Inception Module
    reductions = {
        64,
        64,
        32,
        128
    }
    expansions = {
        256,
        64
    }
    net:add(inceptionModule(128, 512, reductions, expansions))
    net:add(nn.SpatialConvolution(512, 768, 3, 3, 1, 1))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
    -- Inception Module
    reductions = {
        64,
        256,
        256,
        128
    }
    expansions = {
        320,
        512
    }
    net:add(inceptionModule(768, 1024, reductions, expansions))
    net:add(nn.SpatialAveragePooling(5, 5, 1, 1))
    net:add(nn.View(1024))
    net:add(nn.Linear(1024, 512))
    net:add(nn.Dropout(0.4))
    net:add(nn.Linear(512, 256))
    net:add(nn.Dropout(0.4))
    net:add(nn.Linear(256, 10))
    print(net)
    return net
end

-------------------------------------------------------------------------------
------------------------VISUALIZATION FUNCTIONS--------------------------------
-------------------------------------------------------------------------------
-- Load a pre-trained model, remove all SpatialBatchNormalization layers
-- SpatialBatchNormalization requires batches of images, visualization 
-- explicitly uses only a single image
-- There may be other layer types that have this problem that I am unaware of
local function loadModel()
    net = torch.load('model-nets/model--float.net')
    for i, module in ipairs(net.modules) do
        if torch.type(module) == 'nn.SpatialBatchNormalization' then
            net:remove(i)
        end
    end
    print(net)
    return net
end

-- Retrieve the filter responses caused by passing the image through the model
-- Each table of filter responses from a layer has a field 'ADDED_NAME' 
-- added to it which contains the name of the layer type. This is to make
-- reviewing filter responses and mapping them back to layers easier...
-- Return the filter responses in a table 
local function getLayerResponses(model, image)
    model:evaluate()
    model:forward(image)
    filterResponses = {}
    for i, curModule in ipairs(model.modules) do
        local activation = curModule.output.new()
        activation:resize(curModule.output:nElement())
        activation:copy(curModule.output)
        curModule['ADDED_NAME'] = torch.type(curModule)
        table.insert(filterResponses, curModule)
    end
    print(filterResponses)
    return filterResponses
end

-- Generate gnuplots from the passed in table of filter responses
-- place individual folders which will hold the filter responses into the
-- directory specified by objectName
local function visualizeFilterResponses(filterResponses, originalImage, objectName) 
    local DIRECT_DISPLAY = false
    if not DIRECT_DISPLAY and not objectName then
        print('\n\n\n')
        print('YOU MUST INCLUDE AN OBJECT NAME IF YOU ARE SAVING TO FILE!!!!')
        return
    end
    local figLayerId = 10
    for k, v in ipairs(filterResponses) do
        print("Number of Filter responses for this layer: ")
        print(v.output:size(2))
        for i = 1, v.output:size(2) do
            -- Direct Display
            if DIRECT_DISPLAY then
                gnuplot.figure(k * figLayerId + i);
                gnuplot.imagesc(v.output[1][i])
            -- Save To File
            else
                local objectFolderName = 'object-' .. objectName
                objectFolderName = paths.concat(filterResponsesFolderName, objectFolderName)
                local layerFolderName = 'layer-' .. tostring(k)
                layerFolderName = paths.concat(objectFolderName, layerFolderName)
                local filePath = paths.concat(layerFolderName, 'filter-' .. tostring(i))

                if not paths.dir(objectFolderName) then
                    paths.mkdir(objectFolderName)
                end
                if not paths.dir(layerFolderName) then
                    paths.mkdir(layerFolderName)
                end

                gnuplot.pngfigure(filePath)
                gnuplot.imagesc(v.output[1][i])
                gnuplot.plotflush()
            end

            if i > 5 then
                break
            end
        end
        break
        figLayerId = figLayerId * 10
    end
    gnuplot.figure(1)
    gnuplot.imagesc(originalImage[1][1])
    gnuplot.figure(2)
    gnuplot.imagesc(originalImage[1][2])
    gnuplot.figure(3)
    gnuplot.imagesc(originalImage[1][3])
end

-- Close all gnuplot windows
local function clearGnuPlots()
    os.execute('pkill gnuplot')
end


-------------------------------------------------------------------------------
-----------------------------MAIN RUNNER CODE----------------------------------
-------------------------------------------------------------------------------

clearGnuPlots()

-- Flip between model loaded from file and secondArch defined above in function
local USE_LOADED_MODEL = false

-- Load image
local fileName = paths.concat(imageFolderName, '1.png')
testImage = image.load(fileName, 3, 'float')
testImage = testImage:reshape(1, 3, 32, 32)
print("Input to model dimensions: ")
print(testImage:size())
-- Load model from file or use test model to validate
if USE_LOADED_MODEL then
    model = loadModel()
else
    testImage = testImage:type('torch.DoubleTensor')
    model = secondArch()
end

-- Get filter responses and visualize
filterResponses = getLayerResponses(model, testImage)
visualizeFilterResponses(filterResponses, testImage, 'frog-1')
