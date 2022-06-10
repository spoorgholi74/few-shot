import torch
import torch.optim as optim

from helpers import *
from model import load_protonet_conv
from train import train

if __name__ == '__main__':
    # Check GPU support, please do activate GPU
    print(torch.cuda.is_available())

    # Create the training and testing dataset
    trainx, trainy = read_images('images_background')
    # testx, testy = read_images('images_evaluation')

    # Create a sample and display it
    sample_example = extract_sample(8, 5, 5, trainx, trainy)
    display_sample(sample_example['images'])

    model = load_protonet_conv(
        x_dim=(3, 28, 28),
        hid_dim=64,
        z_dim=64,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_way = 60
    n_support = 5
    n_query = 5

    train_x = trainx
    train_y = trainy

    max_epoch = 5
    epoch_size = 2000

    # Train model
    train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size)

    # Test model
    n_way = 5
    n_support = 5
    n_query = 5

    # test_x = testx
    # test_y = testy

    test_episode = 1000

    # test(model, test_x, test_y, n_way, n_support, n_query, test_episode)
    # my_sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
    # display_sample(my_sample['images'])

    # my_loss, my_output = model.set_forward_loss(my_sample)
    # print(my_output)
