"""
utils for custom dataset
"""
import os
import sys
import pandas as pd

import numpy
import gc

import theano
import theano.tensor as T

      
def load_data(dataset, nfold=5):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'

    # Load the dataset
    
    ###################################################
    # pcs, 20150731, add support for multiple dataset, use a function to do this to decrease memory usage
    # command line usage:   python DBN_pcs.py dataset1.pkl,dataset2.pkl,dataset3.pkl,dataset4.pkl 2000 3
    # df = pd.read_pickle(dataset) # Original code
    def load_pickle(dataset):
        file_list = dataset.split(',')
        data_list = []
        for file in file_list:
            print 'Load Data set:', file
            #data_list.append( pd.read_pickle(file) )
            if os.path.splitext(file)[-1] == '.pkl':
                data_list.append( pd.read_pickle(file) )
            elif os.path.splitext(file)[-1] == '.tsv':
                data_list.append( pd.read_table(file, dtype=numpy.float32) )
            elif os.path.splitext(file)[-1] == '.hdfs':
                hdf = pd.HDFStore(file)
                data_list.append( hdf['data'] )
            elif os.path.splitext(file)[-1] == '.npy':
                #data_list.append( numpy.load(file) )
                data_list.append( numpy.asarray( numpy.load(file), order='C' ) )
            else:
                pass
        if len(data_list) == 1:
            df = data_list[0]
        else:
            if type(data_list[0]) != type(numpy.zeros(0)):
                df = pd.concat ( data_list, ignore_index=True )
            else:
                df = numpy.concatenate( data_list )
        #print type(df), df.shape
        return df

    df = load_pickle(dataset)
    ###################################################

    #if os.path.basename(dataset) != 'mnist':
    if os.path.basename(dataset) != 'mnist' and  type(df) != type(numpy.zeros(0)):
        numpy.random.seed(123)
        print 'Data set permutation'
        df = df.reindex(numpy.random.permutation(df.index))
        
        # purge invariance
        print 'Data purge invariance'
        df = df.ix[:, df.max() != df.min()]

        gc.collect()

        # scaling [0,1]
        print 'Data scaling'
        # split the following line for lower memory usage
        #df.ix[:,:-1] = (df.ix[:,:-1] - df.ix[:,:-1].min()) / (df.ix[:,:-1].max() - df.ix[:,:-1].min())
        max = df.ix[:,:-1].max()
        min = df.ix[:,:-1].min()
        diff = max - min
        df.ix[:,:-1] = (df.ix[:,:-1] - min) / diff

    ###################################################
    # jason, 20150709, pcs, the following temp code are used to change column size to test input vector size other than 1863, note: pandas matrix index start from 1 not 0
    #df = df.ix[:, 1000:]
    #import pickle
    #pickle.dump(df, open('dataset_preprocessed.pkl', 'wb'), pickle.HIGHEST_PROTOCOL )
    #sys.exit(0)
    print "Data shape after preprocessing:", df.shape
    ###################################################

    train_set = df[:-df.shape[0] / nfold]
    test_set = df[-df.shape[0] / nfold:]

    #train_set, test_set format: pandas.DataFrame(columns=[0,1,2,...,label])

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        #data_x = data_xy.ix[:,:-1]
        #data_y = data_xy.ix[:,-1]
        if type(data_xy) != type(numpy.zeros(0)):
            data_x = data_xy.ix[:,:-1]
            data_y = data_xy.ix[:,-1]
        else:
            data_x = data_xy[:,:-1]
            data_y = data_xy[:,-1]

        ###################################################
        # jason, 20150709, pcs, add order='C' to force row-major storage for better performance
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX, order='C'),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX, order='C'),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returnin
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    gc.collect()
    return rval


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            numpy.resize(this_x, img_shape))
                    else:
                        this_img = numpy.resize(this_x, img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

if __name__ == '__main__':
    DATASET = 'data/gpcr'
    print(load_data(DATASET))
